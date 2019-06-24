from settings import *
from keras import Model, Input
from keras.applications import MobileNet, DenseNet121, ResNet50
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam



class DeepMN:

    def __init__(self, classes=None, weights='imagenet', dropout_global=1e-3,
                 dropout=0.6, activation='relu',net_type='mobnet', loss='categorical_crossentropy', eta=0.001,
                 amsgrad=False, *args, **kwargs):
        self.net_type=net_type
        self.classes = len(classes)
        self.w = weights
        self.dropout_global = dropout_global
        self.dropout = dropout
        self.activation = activation
        self.loss = loss
        self.eta = eta
        self.amsgrad = amsgrad

    def create_model(self):
        from helper import top_3_categorical_acc, precision, recall, f1_score

        inputs = Input(shape=dim, name='input')
        assert self.net_type  in ['mobnet', 'densenet', 'resnet'], 'Please specify the correct name of architecture from the following : {}'.format(
            ['mobnet', 'densenet', 'resnet'])
        if self.net_type=='mobnet':
            model_mobilenet = MobileNet(input_shape=dim, alpha=1, depth_multiplier=1, dropout=self.dropout_global,
                                        include_top=False, weights=self.w, input_tensor=None)

            x = model_mobilenet(inputs)

            x = Dense(1024, activation='relu')(x)
            x = Dropout(self.dropout)(x)
            x = Dense(512, activation='relu')(x)

            x = GlobalAveragePooling2D()(x)
        elif self.net_type=='densenet':
            dense_net=DenseNet121(include_top=False,
                                  weights=self.w, input_tensor=None, input_shape=dim)
            x=dense_net(inputs)
            x=GlobalAveragePooling2D()(x)
            x=Dropout(self.dropout)(x)
        elif self.net_type=='resnet':
            resnet=ResNet50(include_top=False, weights=self.w, input_tensor=None,
                                               input_shape=dim)
            x=resnet(inputs)
            x=GlobalAveragePooling2D()(x)
            x=Dropout(self.dropout)(x)

        z = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=z)
        adam = Adam(lr=self.eta, amsgrad=self.amsgrad)

        model.compile(optimizer=adam, loss=self.loss, metrics=['accuracy', top_3_categorical_acc,

                                                               precision, recall, f1_score])

        print(model.summary())

        return model, adam
