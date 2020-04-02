from tensorflow.keras.layers import Layer
from roi.roi_tf import roi_pool_tf


class RoipoolingLayer(Layer):

    def __init__(self, **kwargs):
        super(RoipoolingLayer, self).__init__(**kwargs)

    def call(self, inputs, rois, im_size):
        output = roi_pool_tf([inputs, rois], im_size)
        return output
