import argparse
import os
import cv2
from utils.test_utils import preprocess_test_img, bbox_reg_target
from models.model_body import get_model_body
from models.rpn_model import get_rpn_model
from models.fast_rcnn_model import get_fastrcnn_model
from keras.layers import Input
from config import Config
from exception import ValueEmptyException
from rpn.proposal_layer import proposal_layer
import numpy as np
from utils.nms import py_cpu_nms
from utils.softmax import softmax

config = Config()

# load config
img_widht, img_height = config.im_size
anchors_num = config.anchors_num
classes_num = config.classes_num
keep_prob = config.keep_prob
feat_stride = config.feat_stride
thresh = config.test_rpn_nms_thresh

class_names = [name.strip() for name in open('./data/voc_classes.txt').readlines()]
class_names = ['bg'] + class_names
class_mapping = {v: k for k, v in enumerate(class_names)}

class_color_mapping = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


def _main(args):
    img_path = os.path.expanduser(args.img_path)
    file_name = os.path.basename(img_path)
    net = args.net
    output_path = args.output
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    img = cv2.imread(img_path)

    X, ratios = preprocess_test_img(img, config)

    # 输入tensor
    input_tensor = Input(shape=(img_height, img_widht, 3))
    input_rois = Input(shape=(None, 5))

    # shares conv
    model_body = get_model_body(input_tensor, net=net)

    # rpn
    rpn_model = get_rpn_model(model_body, anchors_num)
    # fastrcnn
    fastrcnn_model = get_fastrcnn_model(model_body=model_body, input_rois=input_rois,
                                        classes_num=classes_num, keep_prob=keep_prob)
    # load weights
    if not os.path.exists(args.rpn_model_weights_path) or \
            not os.path.exists(args.fastrccn_model_weights_path):
        raise ValueEmptyException('rpn_model_weights_path or fastrccn_model_weights_path is null, please check it')
    rpn_model.load_weights(args.rpn_model_weights_path, by_name=True)
    fastrcnn_model.load_weights(args.fastrccn_model_weights_path, by_name=True)

    rpn_bbox_cls, rpn_bbox_pred = rpn_model.predict(X)
    # shape (None, 5)   (batch_id, x1, y1, x2, y2)
    rois = proposal_layer(rpn_bbox_cls, rpn_bbox_pred, (img_widht, img_height), feat_stride, eval_mode=True)
    rois = np.expand_dims(rois, axis=0)
    # (1, 300, 21)  (1, 300, 80)
    fastrcnn_cls_output, fastrcnn_reg_output = fastrcnn_model.predict([X, rois])
    # (300, 21)
    fastrcnn_cls_output = np.squeeze(fastrcnn_cls_output, axis=0)
    fastrcnn_cls_output = softmax(fastrcnn_cls_output)
    # (300, 80)
    fastrcnn_reg_output = np.squeeze(fastrcnn_reg_output, axis=0)
    # 最大值索引 (300,)
    argmax_cls = np.argmax(fastrcnn_cls_output, axis=1)
    # 取出最大类别 (300,)
    max_cls = fastrcnn_cls_output[np.arange(len(argmax_cls)), argmax_cls]
    # (None, 6) -- x1, y1, x2, y2, score, cls
    pred_boxes = bbox_reg_target(fastrcnn_reg_output, argmax_cls, rois, max_cls)
    pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]] / ratios[0]
    pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]] / ratios[1]

    # 非极大值抑制抑制
    keep_ind = py_cpu_nms(pred_boxes, thresh)
    final_boxes = pred_boxes[keep_ind, :]
    # draw_rect
    # final_boxes = [[1, 1, 100, 100, 0.9, 1]]
    for idx in range(len(final_boxes)):
        # x1, y1, x2, y2, score, cls
        x1, y1, x2, y2, score, cls = final_boxes[idx]
        # ratios 缩放回原图大小
        x1, x2 = int(round(x1 / ratios[0])), int(round(x2 / ratios[0]))
        y1, y2 = int(round(y1 / ratios[1])), int(round(y2 / ratios[1]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (int(class_color_mapping.get(cls)[0]),
                                                int(class_color_mapping.get(cls)[1]),
                                                int(class_color_mapping.get(cls)[2])), 2)
        text_lable = '%s: %s' % (class_names[cls], score)
        textOrg = (x1, y1 - 2)
        cv2.putText(img, text_lable, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(output_path, file_name), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='faster-rcnn networks detection argument')
    parser.add_argument('--rpn_model_weights_path', default='./logs/rpn_model_weights.h5',
                        help='rpn model weights path')
    parser.add_argument('--fastrccn_model_weights_path', default='./logs/fastrcnn_model_weights.h5',
                        help='fastrcnn model path')
    parser.add_argument('--img_path', default='', help='检测图片路径')
    parser.add_argument('--net', default='vgg16', help='获取共享卷积层使用网络')
    parser.add_argument('--output', default='./test/', help='检测结果输出地址')
    args = parser.parse_args()
    _main(args)
