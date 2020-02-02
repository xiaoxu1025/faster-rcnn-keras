from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from config import Config
from exception import ValueValidException

config = Config()


def model_body(net='vgg16', trainable=False):
    if net == config.network[0]:
        pre_model = VGG16(include_top=False)
    elif net == config.network[1]:
        pre_model = ResNet50(include_top=False)
    elif net == config.network[2]:
        pre_model = InceptionV3(include_top=False)
    else:
        raise ValueValidException('%s not defined yet, please choose another' % net)
    # 默认是不可以训练的  如果需要训练可以放开
    if not trainable:
        for layer in pre_model.layers:
            layer.trainable = False
    model = Model(inputs=pre_model.input, outputs=pre_model.get_layer('block5_conv3').output)
    return model
