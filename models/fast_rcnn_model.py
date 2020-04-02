from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Lambda
from roi.roi_tf import roi_pool_tf
import keras.backend as K
from models.roi_pooling_layer import RoipoolingLayer


def get_fastrcnn_model(model_body, input_rois, classes_num=20,
                       keep_prob=.5, im_size=(800, 608), **kwargs):
    share_features = model_body.output
    output = RoipoolingLayer(name='roipooling')(share_features, input_rois)
    output = Flatten(name='flatten')(output)
    output = Dense(4096, activation='relu', name='fc1')(output)
    output = Dropout(rate=keep_prob)(output)
    output = Dense(4096, activation='relu', name='fc2')(output)
    output = Dropout(rate=keep_prob)(output)
    # 最后输出扩充一个batch_size维度
    fastrcnn_cls_output = Dense(classes_num + 1, activation='softmax',
                                kernel_initializer='zero', name='fast_rcnn_cls')(output)
    fastrcnn_reg_output = Dense(classes_num * 4, activation='softmax',
                                kernel_initializer='zero', name='fast_rcnn_reg')(output)
    # fastrcnn_cls_output, fastrcnn_reg_output = Lambda(expand_dims, name='expand_output_dims')([fastrcnn_cls_output,
    #                                                                                            fastrcnn_reg_output])
    fastrcnn_cls_output = Lambda(expand_dims, name='expand_cls_output_dims')(fastrcnn_cls_output)
    fastrcnn_reg_output = Lambda(expand_dims, name='expand_reg_output_dims')(fastrcnn_reg_output)
    # Model的ouputs 必须是某一层的输出
    fastrcnn_model = Model([model_body.input, input_rois], [fastrcnn_cls_output, fastrcnn_reg_output])
    return fastrcnn_model


def expand_dims(args):
    return K.expand_dims(args, axis=0)

# def expand_dims(args):
#     return [K.expand_dims(args, axis=0), K.expand_dims(args[1], axis=0)]

# from models.model_body import get_model_body
# from keras.layers import Input
#
# img_input = Input(shape=(608, 800, 3))
# model_body = get_model_body(img_input, 'vgg16')
# rois = Input(shape=(None, 5))
# fastrcnn_model = get_fastrcnn_model(model_body, rois)
# fastrcnn_model.summary()
