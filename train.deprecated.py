# from keras.optimizers import Adam
# from voc_data import VocData
# from models.rnp import RPN
# from models.fast_rcnn_model import FastRCNN
# import os
# from keras.callbacks import TensorBoard
# from keras.utils import generic_utils
# import numpy as np
# import time
# import tensorflow as tf
# from keras import Sequential
# from rpn.proposal_target_layer import proposal_target_layer
# from rpn.proposal_layer import proposal_layer
# from config import Config
# from loss_func import rpn_cls_loss, rpn_reg_loss, fastrcnn_cls_loss, fastrcnn_reg_loss
#
# cfg = Config()
#
# batch_size = 1
# voc_train_data = VocData('~/segment_data', 2007, 'train', './data/voc_classes.txt')
# voc_train_g = voc_train_data.data_generator_wrapper(batch_size)
# voc_val_data = VocData('~/segment_data', 2007, 'val', './data/voc_classes.txt')
# voc_val_g = voc_val_data.data_generator_wrapper(batch_size)
#
# rpn_model = RPN(anchors_num=9, net='vgg16', model_body_trainable=True, eval_mode=False)
# # 得到共享feature_maps
# # 继承子类api实现的model 貌似没有input 无法使用Model(rpn_model.input, rpn_model.output)
# model_body = Sequential()
# model_body.add(rpn_model.get_layer(index=0))
# for layer in rpn_model.get_layer('model_1').layers:
#     model_body.add(layer)
#
# fastrcnn_model = FastRCNN(voc_train_data.classes_num)
#
# optimizer = Adam(lr=1e-5)
# optimizer_classifier = Adam(lr=1e-5)
# # 这里不能设置metrics 因为我的labels格式和输出格式不匹配 会报错
# rpn_model.compile(optimizer=optimizer,
#                   loss=[rpn_cls_loss, rpn_reg_loss])
# fastrcnn_model.compile(optimizer=optimizer_classifier,
#                        loss=[fastrcnn_cls_loss, fastrcnn_reg_loss])
# log_path = './logs'
# if not os.path.exists(log_path):
#     os.mkdir(log_path)
# rpn_callback = TensorBoard(os.path.join(log_path, '000'))
# rpn_callback.set_model(rpn_model)
#
# fastrcnn_callback = TensorBoard(os.path.join(log_path, '111'))
# fastrcnn_callback.set_model(fastrcnn_model)
#
# epoch_length = voc_train_data.sample_nums
# num_epochs = 100
# train_step = 0
# iter_num = 0
# losses = np.zeros((epoch_length, 4))
#
# best_loss = np.Inf
#
#
# # tensorboard
# def write_log(callback, names, logs, batch_no):
#     for name, value in zip(names, logs):
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         callback.writer.add_summary(summary, batch_no)
#         callback.writer.flush()
#
#
# for epoch_num in range(num_epochs):
#     start_time = time.time()
#     progbar = generic_utils.Progbar(epoch_length)  # keras progress bar 사용
#     print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
#     while True:
#         # (1, 342, 50)
#         # (1, 38, 50, 108)
#         X, Y, gt_boxes = next(voc_train_g)
#         # 去掉gt_boxes第一个维度
#         gt_boxes = np.squeeze(gt_boxes, axis=0)
#         rpn_loss = rpn_model.train_on_batch(X, Y)
#         # write_log(callback, ['Elapsed time', 'rpn_loss'], [time.time() - start_time, rpn_loss], train_step)
#         train_step += 1
#         # 这里计算出三个loss
#         losses[iter_num, [0, 1]] = rpn_loss[0:2]
#
#         iter_num += 1
#
#         # 训练fast_rcnn
#         rpn_bbox_cls, rpn_bbox_pred = rpn_model.predict_on_batch(X)
#         # get shared feature_maps
#         feature_maps = model_body.predict_on_batch(X)
#         # get fastrcnn_model 的训练数据集
#         rois = proposal_layer(rpn_bbox_cls, rpn_bbox_pred, cfg.im_size, cfg.feat_stride, eval_mode=False)
#         train_rois, labels, bbox_targets = proposal_target_layer(rois, gt_boxes, voc_train_data.classes_num)
#         # 添加batch_size 维度
#         train_rois = train_rois.reshape((1,) + train_rois.shape)
#         fastrcnn_loss = fastrcnn_model.train_on_batch([feature_maps, train_rois], [labels, bbox_targets])
#         losses[iter_num, [2, 3]] = fastrcnn_loss[0:2]
#
#         progbar.update(iter_num, [('rpn_cls_loss', np.mean(losses[:iter_num, 0])),
#                                   ('rpn_reg_loss', np.mean(losses[:iter_num, 1])),
#                                   ('rpn_total_loss', np.mean(np.sum(losses[:iter_num, [0, 1]], axis=-1), axis=0)),
#                                   ('fastrcnn_cls_loss', np.mean(losses[:iter_num, 2])),
#                                   ('fastrcnn_reg_loss', np.mean(losses[:iter_num, 3])),
#                                   ('fastrcnn_total_loss', np.mean(np.sum(losses[:iter_num, [2, 3]], axis=-1), axis=0)),
#                                   ])
#         if iter_num == epoch_length:
#             # 停止一个循环
#             loss_rpn_cls = np.mean(losses[:iter_num, 0])
#             loss_rpn_reg = np.mean(losses[:iter_num, 1])
#             fastrcnn_cls_loss = np.mean(losses[:iter_num, 2])
#             fastrcnn_reg_loss = np.mean(losses[:iter_num, 3])
#             curr_loss = loss_rpn_cls + loss_rpn_reg + fastrcnn_cls_loss + fastrcnn_reg_loss
#             # write_log(callback, ['Elapsed time', 'mean_loss_rpn_cls', 'curr_loss',
#             #                      'loss_rpn_cls', 'loss_rpn_reg', 'fastrcnn_cls_loss', 'fastrcnn_reg_loss'],
#             #           [time.time() - start_time, loss_rpn_cls, curr_loss,
#             #            loss_rpn_cls, loss_rpn_reg, fastrcnn_cls_loss, fastrcnn_reg_loss], epoch_num + 1)
#             print('curr_loss: {}'.format(curr_loss))
#             print('mean_loss_rpn_cls: {}'.format(loss_rpn_cls))
#             print('mean_loss_rpn_reg: {}'.format(loss_rpn_reg))
#             print('mean_loss_fastrcnn_cls: {}'.format(fastrcnn_cls_loss))
#             print('mean_loss_fastrcnn_reg: {}'.format(fastrcnn_reg_loss))
#             print('epoch: {} Elapsed time: {}'.format(epoch_num + 1, time.time() - start_time))
#             if curr_loss < best_loss:
#                 best_loss = curr_loss
#                 rpn_model.save_weights(os.path.join(log_path, 'rpn_model_weights.h5'))
#                 fastrcnn_model.save_weights(os.path.join(log_path, 'fastrcnn_model_weights.h5'))
#             iter_num = 0
#             break
