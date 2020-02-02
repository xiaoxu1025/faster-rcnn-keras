from keras.layers import Layer
import tensorflow as tf
from roi.roi_tf import roi_pool_tf
from keras.layers import Lambda


class RoiPoolingConv(Layer):

    def __init__(self, pooled_height, pooled_width, im_size):
        super(RoiPoolingConv, self).__init__()
        self._pooled_height = pooled_height
        self._pooled_width = pooled_width
        self._im_size = im_size
        self._roipooling = Lambda(roi_pool_tf, arguments={'im_dims': im_size}, name='roipooling')

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2, 'RoiPoolingConv inputs len should be 2 [feature_maps, rois]'
        feature_maps = inputs[0]
        rois = inputs[1]
        # output = roi_pool_tf(feature_maps, rois, self._im_size)
        output = self._roipooling([feature_maps, rois])
        # 增加batch_size维度
        output = tf.reshape(output, (1,) + output.shape)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((None, None, 7, 7, 512))
