"""
获取rpn_model
"""
from keras import Model
from keras.layers import Conv2D
from models.model_body import model_body
import tensorflow as tf
from keras.layers import ZeroPadding2D


class RPN(Model):

    def __init__(self, anchors_num=9, net='vgg16', model_body_trainable=False, eval_mode=False, **kwargs):
        """

        :param anchors_num: 特征图上每个点预测anchors_num个anchor
        :param net: model body 采用的网络
        :param model_body_trainable: model body是否可训练
        :param eval_mode 验证模式
        """
        super(RPN, self).__init__(**kwargs)
        self._anchors_num = anchors_num
        # 这一层没有意思就是用来当输入层
        self._zero_padding = ZeroPadding2D(padding=(4, 0), name='zero_padding')
        self._model_body = model_body(net, model_body_trainable)
        # 首先用3*3的卷积核融合周围3x3的空间信息
        self._conv2d_01 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='normal',
                                 name='merge_conv2d')
        self._conv2d_02 = Conv2D(anchors_num * 2, 1, activation='sigmoid', kernel_initializer='uniform',
                                 name='cls_conv2d')
        self._conv2d_03 = Conv2D(anchors_num * 4, 1, activation='linear', kernel_initializer='zero',
                                 name='reg_conv2d')

    def call(self, inputs, mask=None, **kwargs):
        inputs = self._zero_padding(inputs)
        feature_maps = self._model_body(inputs)
        x = self._conv2d_01(feature_maps)
        cls_output = self._conv2d_02(x)
        reg_output = self._conv2d_03(x)
        # [(?, 38, 50, 18)(?, 38, 50, 36)]
        return cls_output, reg_output

    def compute_output_shape(self, input_shape):
        return [tf.TensorShape((None, None, None, 18)),
                tf.TensorShape((None, None, None, 36))]
