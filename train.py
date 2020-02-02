from keras.optimizers import Adam
from voc_data import VocData
import os
from keras.callbacks import TensorBoard
from keras.utils import generic_utils
import numpy as np
import time
import tensorflow as tf
from rpn.proposal_target_layer import proposal_target_layer
from rpn.proposal_layer import proposal_layer
from config import Config
from loss_func import rpn_cls_loss, rpn_reg_loss, fastrcnn_cls_loss, fastrcnn_reg_loss
from models.model_body import get_model_body
from models.rpn_model import get_rpn_model
from models.fast_rcnn_model import get_fastrcnn_model
from models.faster_rcnn_model import get_fasterrcnn_model
from keras.layers import Input

cfg = Config()
# 配置参数
img_widht, img_height = cfg.im_size
anchors_num = cfg.anchors_num
classes_num = cfg.classes_num
keep_prob = cfg.keep_prob
pooled_height = cfg.pooled_height
pooled_width = cfg.pooled_width
im_size = cfg.im_size
rois_num = cfg.train_rpn_post_nms_top_n

batch_size = 1
# ~/segment_data 是存放数据地址
# 比如我的数据地址 ~/segment_data/VOCdevkit/VOC2007  这里只需要截取到VOCdevkit上一层即可
voc_train_data = VocData('~/segment_data', 2007, 'train', './data/voc_classes.txt')
voc_train_g = voc_train_data.data_generator_wrapper(batch_size)
voc_val_data = VocData('~/segment_data', 2007, 'val', './data/voc_classes.txt')
voc_val_g = voc_val_data.data_generator_wrapper(batch_size)

# 输入tensor
input_tensor = Input(shape=(img_height, img_widht, 3))
input_rois = Input(shape=(None, 5))

# shares conv
model_body = get_model_body(input_tensor)

# rpn_model
rpn_model = get_rpn_model(model_body, anchors_num)
# fastrcnn_model
fastrcnn_model = get_fastrcnn_model(model_body=model_body, input_rois=input_rois,
                                    classes_num=classes_num, keep_prob=keep_prob)
# fasterrcnn_model
fasterrcnn_model = get_fasterrcnn_model(model_body=model_body, input_rois=input_rois,
                                        rpn_model=rpn_model, fastrcnn_model=fastrcnn_model)

if os.path.exists('./logs/fastrcnn_model_weights.h5'):
    fastrcnn_model.load_weights('./logs/fastrcnn_model_weights.h5', by_name=True)

if os.path.exists('./logs/rpn_model_weights.h5'):
    rpn_model.load_weights('./logs/rpn_model_weights.h5', by_name=True)

if os.path.exists('./logs/fasterrcnn_model_weights.h5'):
    fastrcnn_model.load_weights('./logs/fasterrcnn_model_weights.h5', by_name=True)

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
rpn_model.compile(optimizer=optimizer,
                  loss=[rpn_cls_loss, rpn_reg_loss])
fastrcnn_model.compile(optimizer=optimizer_classifier,
                       loss=[fastrcnn_cls_loss, fastrcnn_reg_loss])
# loss 可以随便写 不需要训练
# fasterrcnn_model.compile(optimizer='sgd', loss=lambda y_true, y_pred: tf.zeros(1))

log_path = './logs'
if not os.path.exists(log_path):
    os.mkdir(log_path)
rpn_callback = TensorBoard(os.path.join(log_path, '000'))
rpn_callback.set_model(rpn_model)

fastrcnn_callback = TensorBoard(os.path.join(log_path, '111'))
fastrcnn_callback.set_model(fastrcnn_model)

fasterrcnn_callback = TensorBoard(os.path.join(log_path, '222'))
fasterrcnn_callback.set_model(fasterrcnn_model)

# 训练参数
epoch_length = voc_train_data.sample_nums
num_epochs = 10
train_step = 0
iter_num = 0
losses = np.zeros((epoch_length, 4))

best_loss = np.Inf


# tensorboard
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def save_weights(rpn_model, fastrcnn_model, fasterrcnn_model):
    rpn_model_path = os.path.join(log_path, 'rpn_model_weights.h5')
    fastrcnn_model_path = os.path.join(log_path, 'fastrcnn_model_weights.h5')
    fasterrcnn_model_path = os.path.join(log_path, 'fasterrcnn_model_weights.h5')
    if os.path.exists(rpn_model_path):
        os.remove(rpn_model_path)
    if os.path.exists(rpn_model_path):
        os.remove(fastrcnn_model_path)
    if os.path.exists(rpn_model_path):
        os.remove(fasterrcnn_model_path)
    rpn_model.save_weights(rpn_model_path)
    fastrcnn_model.save_weights(fastrcnn_model_path)
    fasterrcnn_model.save_weights(fasterrcnn_model_path)


for epoch_num in range(num_epochs):
    start_time = time.time()
    progbar = generic_utils.Progbar(epoch_length)  # keras progress bar
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    while True:
        # (1, 342, 50)
        # (1, 38, 50, 108)
        X, Y, gt_boxes = next(voc_train_g)
        # 去掉gt_boxes第一个维度
        gt_boxes = np.squeeze(gt_boxes, axis=0)
        rpn_loss = rpn_model.train_on_batch(X, Y)

        write_log(rpn_callback, ['Elapsed time', 'rpn_cls_loss', 'rpn_reg_loss'],
                  [time.time() - start_time, rpn_loss[0], rpn_loss[1]], train_step)

        train_step += 1
        # 这里计算出三个loss
        losses[iter_num, [0, 1]] = rpn_loss[0:2]

        iter_num += 1

        # 训练fast_rcnn
        rpn_bbox_cls, rpn_bbox_pred = rpn_model.predict_on_batch(X)
        # get shared feature_maps
        # get fastrcnn_model 的训练数据集
        rois = proposal_layer(rpn_bbox_cls, rpn_bbox_pred, cfg.im_size, cfg.feat_stride, eval_mode=False)
        train_rois, labels, bbox_targets = proposal_target_layer(rois, gt_boxes, voc_train_data.classes_num)
        # 添加batch_size 维度
        train_rois = train_rois.reshape((1,) + train_rois.shape)
        fastrcnn_loss = fastrcnn_model.train_on_batch([X, train_rois], [labels, bbox_targets])

        write_log(fastrcnn_callback, ['Elapsed time', 'fastrcnn_cls_loss', 'fastrcnn_reg_loss'],
                  [time.time() - start_time, fastrcnn_loss[0], fastrcnn_loss[1]], train_step)

        losses[iter_num, [2, 3]] = fastrcnn_loss[0:2]

        train_step += 1

        progbar.update(iter_num, [('rpn_cls_loss', np.mean(losses[:iter_num, 0])),
                                  ('rpn_reg_loss', np.mean(losses[:iter_num, 1])),
                                  ('rpn_total_loss', np.mean(np.sum(losses[:iter_num, [0, 1]], axis=-1), axis=0)),
                                  ('fastrcnn_cls_loss', np.mean(losses[:iter_num, 2])),
                                  ('fastrcnn_reg_loss', np.mean(losses[:iter_num, 3])),
                                  ('fastrcnn_total_loss', np.mean(np.sum(losses[:iter_num, [2, 3]], axis=-1), axis=0)),
                                  ])
        if iter_num % 100 == 0:
            # 停止一个循环
            loss_rpn_cls = np.mean(losses[:iter_num, 0])
            loss_rpn_reg = np.mean(losses[:iter_num, 1])
            fastrcnn_cls_loss = np.mean(losses[:iter_num, 2])
            fastrcnn_reg_loss = np.mean(losses[:iter_num, 3])
            curr_loss = loss_rpn_cls + loss_rpn_reg + fastrcnn_cls_loss + fastrcnn_reg_loss
            if curr_loss < best_loss:
                best_loss = curr_loss
                # save weigths
                save_weights(rpn_model, fastrcnn_model, fasterrcnn_model)

        if iter_num == epoch_length:
            # 停止一个循环
            loss_rpn_cls = np.mean(losses[:iter_num, 0])
            loss_rpn_reg = np.mean(losses[:iter_num, 1])
            fastrcnn_cls_loss = np.mean(losses[:iter_num, 2])
            fastrcnn_reg_loss = np.mean(losses[:iter_num, 3])
            curr_loss = loss_rpn_cls + loss_rpn_reg + fastrcnn_cls_loss + fastrcnn_reg_loss
            write_log(fasterrcnn_callback, ['Elapsed time', 'mean_loss_rpn_cls', 'curr_loss',
                                            'loss_rpn_cls', 'loss_rpn_reg', 'fastrcnn_cls_loss', 'fastrcnn_reg_loss'],
                      [time.time() - start_time, loss_rpn_cls, curr_loss,
                       loss_rpn_cls, loss_rpn_reg, fastrcnn_cls_loss, fastrcnn_reg_loss], epoch_num)
            print('curr_loss: {}'.format(curr_loss))
            print('mean_loss_rpn_cls: {}'.format(loss_rpn_cls))
            print('mean_loss_rpn_reg: {}'.format(loss_rpn_reg))
            print('mean_loss_fastrcnn_cls: {}'.format(fastrcnn_cls_loss))
            print('mean_loss_fastrcnn_reg: {}'.format(fastrcnn_reg_loss))
            print('epoch: {} Elapsed time: {}'.format(epoch_num + 1, time.time() - start_time))
            if curr_loss < best_loss:
                best_loss = curr_loss
                # save weigths
                save_weights(rpn_model, fastrcnn_model, fasterrcnn_model)
            iter_num = 0
            break
