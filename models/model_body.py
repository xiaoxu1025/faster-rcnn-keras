from config import Config
from exception import ValueValidException
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
# from keras.layers import Input

config = Config()


def get_model_body(input_tensor, net='vgg16', trainable=True):
    if net == config.network[0]:
        pre_model = VGG16(input_tensor=input_tensor, include_top=False)
    elif net == config.network[1]:
        pre_model = ResNet50(input_tensor=input_tensor, include_top=False)
    elif net == config.network[2]:
        pre_model = InceptionV3(input_tensor=input_tensor, include_top=False)
    else:
        raise ValueValidException('%s not defined yet, please choose another' % net)
    # 默认是不可以训练的  如果需要训练可以放开
    if not trainable:
        for layer in pre_model.layers:
            layer.trainable = False
    model = Model(inputs=pre_model.input, outputs=pre_model.get_layer(index=-2).output)
    return model
