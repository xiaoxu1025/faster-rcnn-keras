from keras import Model
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from config import Config
from models.roi_pooling_layer import RoiPoolingConv
import tensorflow as tf


class FastRCNN(Model):
    def __init__(self, num_classes=20, keep_prob=.5):
        super(FastRCNN, self).__init__()
        self._num_classes = num_classes
        cfg = Config()
        pooled_h, pooled_w, im_size = cfg.pooled_height, cfg.pooled_width, cfg.im_size
        self._pooled_h, self._pooled_w = pooled_h, pooled_w
        self._roipooling = RoiPoolingConv(pooled_h, pooled_w, im_size)
        self._flatten = TimeDistributed(Flatten(name='flatten'))
        self._dense1 = Dense(4096, activation='relu', name='fc1')
        self._dropout1 = Dropout(rate=keep_prob)
        self._dense2 = Dense(4096, activation='relu', name='fc2')
        self._dropout2 = Dropout(rate=keep_prob)
        # 预测K + 1个类别
        self._dense3 = Dense(num_classes + 1, activation='softmax', kernel_initializer='zero', name='fast_rcnn_cls')
        # 为每个类别预测4个回归值
        self._dense4 = Dense(num_classes * 4, activation='linear', kernel_initializer='zero', name='fast_rcnn_reg')

    def call(self, inputs, mask=None):
        assert len(inputs) == 2, 'FastRCNN inputs len should be 2 [feature_maps, rois]'
        feature_maps = inputs[0]
        rois = inputs[1]
        x = self._roipooling([feature_maps, rois])
        x = self._flatten(x)
        x = self._dense1(x)
        x = self._dropout1(x)
        x = self._dense2(x)
        x = self._dropout2(x)
        cls_output = self._dense3(x)
        reg_output = self._dense4(x)
        return cls_output, reg_output

    def compute_output_shape(self, input_shape):
        return [tf.TensorShape((None, None, self._num_classes + 1)),
                tf.TensorShape((None, None, self._num_classes * 4))]
