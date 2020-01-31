from keras.models import Model


def get_fasterrcnn_model(model_body, input_rois, rpn_model, fastrcnn_model):
    model = Model([model_body.input, input_rois], [*rpn_model.output, *fastrcnn_model.output])
    return model

#
# from keras.layers import Input
# from models.model_body import get_model_body
# from models.rpn_model import get_rpn_model
# from models.fast_rcnn_model import get_fastrcnn_model
#
# input_tensor = Input(shape=(608, 800, 3))
# input_rois = Input(shape=(None, 5))
#
# # shares conv
# model_body = get_model_body(input_tensor)
#
# # rpn
# rpn_model = get_rpn_model(model_body, 9)
# # fastrcnn
# fastrcnn_model = get_fastrcnn_model(model_body=model_body, input_rois=input_rois,
#                                     classes_num=20, keep_prob=.5)
# model = get_fasterrcnn_model(model_body, input_rois, rpn_model, fastrcnn_model)
# model.summary()
