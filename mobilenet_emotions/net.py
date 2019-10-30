from settings import *
from keras import Model, Input
from keras.applications import MobileNet
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, Activation, SeparableConv2D
from keras.optimizers import Adam
from keras.regularizers import l2

from settings import *


class DeepMN:

    def __init__(self, classes=None, weights='imagenet', dropout_global=1e-3,
                 dropout=0.6, activation='relu', net_type='mobnet', loss='categorical_crossentropy', eta=0.001,
                 amsgrad=False, train_mode=False, l2_=0, layer_params=1,
                 trainable=None, *args, **kwargs):
        self.regulizer = l2_
        self.train_mode = train_mode
        self.net_type = net_type
        self.classes = len(classes)
        self.w = weights
        self.dropout_global = dropout_global
        self.dropout = dropout
        self.activation = activation
        self.loss = loss
        self.eta = eta
        self.amsgrad = amsgrad
        self.trainable = trainable
        self.layer_params = layer_params

    def create_model(self):
        from helper import precision, recall, f1_score
        input = Input(shape=(dim[0], dim[1], 3), name='input')
        assert self.net_type in ['mobnet',
                                 'small_mobnet'], 'Please specify the correct name of architecture from the following : {}'.format(
            ['mobnet', 'densenet', 'resnet', 'small_mobnet'])
        if self.net_type == 'mobnet':
            model = MobileNet(input_shape=dim, alpha=1, depth_multiplier=1, dropout=self.dropout_global,
                              include_top=False, weights=self.w, input_tensor=None)
            if self.trainable:
                for i in model.layers[:len(model.layers) - self.trainable]:
                    i.trainable = False
            if not self.trainable:
                x = model(input)

            else:
                for i in model.layers:
                    i.trainable = False
                for i in model.layers[len(model.layers) - self.trainable:]:
                    i.trainable = True
                for i in model.layers:
                    print(i.trainable)
                x = model.output
            x = GlobalAveragePooling2D()(x)
            nodes = 2048
            x = Dense(nodes, activation='relu', kernel_regularizer=l2(self.regulizer))(x)
            x = Dropout(self.dropout)(x)
            for i in range(self.layer_params):
                nodes = int(nodes / 2)
                x = Dense(nodes, activation='relu', kernel_regularizer=l2(self.regulizer))(x)
                x = Dropout(self.dropout)(x)




        else:

            model = MobileNet(input_shape=dim, alpha=0.25, depth_multiplier=1,
                              dropout=self.dropout_global,
                              include_top=False, weights=self.w, input_tensor=None)
            for i in model.layers:
                i.trainable = False
            x = model.output

            x = GlobalAveragePooling2D()(x)

            x = Dense(units=512, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.2)(x)

        if self.classes == 2:
            z = Dense(self.classes)(x)
            z = Activation('sigmoid')(z)
        else:

            z = Dense(self.classes)(x)
            z = Activation('softmax')(z)
        if self.trainable:
            input = model.input
        model = Model(inputs=input, outputs=z)
        adam = Adam(lr=self.eta, amsgrad=self.amsgrad)
        if not self.train_mode:
            model.compile(optimizer=adam, loss=self.loss, metrics=['accuracy',
                                                                   precision, recall, f1_score])

        else:
            model.compile(optimizer=adam, loss=self.loss, metrics=['accuracy'])

        print(model.summary())

        return model, adam
