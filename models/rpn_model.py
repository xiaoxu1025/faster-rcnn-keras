from keras.models import Model
from keras.layers import Conv2D


def get_rpn_model(model_body, anchors_num, **kwargs):
    share_features = model_body.output
    output = Conv2D(256, 3, padding='same', activation='relu',
                    kernel_initializer='normal', name='merge_conv2d')(share_features)
    rpn_cls_output = Conv2D(anchors_num * 2, 1, kernel_initializer='uniform',
                            name='rpn_cls_conv2d')(output)
    rpn_reg_output = Conv2D(anchors_num * 4, 1, kernel_initializer='zero',
                            name='rpn_reg_conv2d')(output)
    rpn_model = Model(model_body.input, [rpn_cls_output, rpn_reg_output])
    return rpn_model

# from models.model_body import get_model_body

# input_tensor = Input(shape=(608, 800, 3))
# model_body = get_model_body(input_tensor, 'vgg16')
# # model_body.summary()
# rpn_model = get_rpn_model(model_body, anchors_num=9, aa=11)
# rpn_model.summary()


