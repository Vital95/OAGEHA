from keras.layers import Dropout, Conv2D, Dense, BatchNormalization, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import MobileNet
from settings import *
from keras import Model, Input


class DeepMN:
    def __init__(self, weights='imagenet'):
        self.w = weights

    def __call__(self):
        inputs = Input(shape=dim, name='the_input')
        model_mobilenet = MobileNet(input_shape=dim, alpha=1, depth_multiplier=1, dropout=1e-3,
                                    include_top=False, weights=self.w, input_tensor=None, pooling=None)
        for i in model_mobilenet.layers[:10]:
            i.trainable = True
        feat_a = model_mobilenet(inputs)
        feat_a = GlobalAveragePooling2D()(feat_a)
        feat_a = Dropout(0.5)(feat_a)
        pred_a_softmax = Dense(num_classes1, activation='softmax', name='age')(feat_a)
        pred_g_softmax = Dense(2, activation='softmax', name='gender')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_g_softmax, pred_a_softmax])

        return model


if __name__ == '__main__':
    net = DeepMN()
