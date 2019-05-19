from keras import Model, Input
from keras.applications import MobileNet
from keras.layers import Dropout, Dense, GlobalAveragePooling2D

from settings import *


class DeepMN:
    def __init__(self, classes=None, weights='imagenet', *args, **kwargs):
        self.classes = len(classes)
        self.w = weights

    def create_model(self):
        inputs = Input(shape=dim, name='input')
        model_mobilenet = MobileNet(input_shape=dim, alpha=1, depth_multiplier=1, dropout=1e-3,
                                    include_top=False, weights=self.w, input_tensor=None)

        x = model_mobilenet(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.6)(x)
        z = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=z)

        return model
