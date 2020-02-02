from keras.models import Model
from models.rnp import RPN
from models.fast_rcnn_model import FastRCNN
import tensorflow as tf


class FasterRCNN(Model):
    def __init__(self, anchors_num=9, net='vgg16', num_classes=20,
                 model_body_trainable=False, keep_prob=0.5, **kwargs):
        super(FasterRCNN, self).__init__(**kwargs)
        self._num_classes = num_classes

        self._rpn = RPN(anchors_num, net, model_body_trainable, **kwargs)
        self._fastrcnn = FastRCNN(num_classes, keep_prob=keep_prob)

    def call(self, inputs, mask=None):
        img_data = inputs[0]
        feature_maps = inputs[2]
        _, _ = self._rpn(img_data)
        rois = self._rpn.rois
        output = self._fastrcnn([feature_maps, rois])
        # 返回rois cls reg
        return [*output, rois]

    def compute_output_shape(self, input_shape):
        return [tf.TensorShape((None, None, self._num_classes + 1)),
                tf.TensorShape((None, None, self._num_classes * 4))]
