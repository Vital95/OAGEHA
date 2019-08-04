from keras.metrics import categorical_accuracy
from settings import *
from keras import Model, Input
from keras.applications import MobileNet, DenseNet121, ResNet50
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, BatchNormalization, SeparableConv2D, MaxPooling2D, \
    Activation, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
from keras.losses import categorical_crossentropy as logloss
class DeepMN:

    def __init__(self, classes=None, weights='imagenet', dropout_global=1e-3,
                 dropout=0.6, activation='relu',net_type='mobnet', loss='categorical_crossentropy', eta=0.001,
                 amsgrad=False,train_mode=False,l2_=0,beta_loss=0.1,
                 lambda_const=0.07,t=10, distillation=False, trainable=None, *args, **kwargs):
        self.regulizer=l2_
        self.train_mode=train_mode
        self.net_type=net_type
        self.classes = len(classes)
        self.w = weights
        self.dropout_global = dropout_global
        self.dropout = dropout
        self.activation = activation
        self.loss = loss
        self.eta = eta
        self.amsgrad = amsgrad
        self.trainable = trainable
        self.beta=beta_loss
        self.temperature=t
        self.lambda_const=lambda_const
        self.distillation_mode= distillation

    def loss_regularized(self, y_true, y_pred):
        entropy = -K.mean(K.sum(y_pred * K.log(y_pred), 1))
        beta = self.beta
        return logloss(y_true, y_pred) - beta * entropy

    def accuracy(self,y_true, y_pred):
        y_true = y_true[:, :self.classes]
        y_pred = y_pred[:, :self.classes]
        return categorical_accuracy(y_true, y_pred)

    def knowledge_distillation_loss(self,y_true, y_pred, lambda_const,temperature):

        # split in
        #    onehot hard true targets
        #    logits from xception
        y_true, logits = y_true[:, :self.classes], y_true[:, self.classes:]

        # convert logits to soft targets
        y_soft = K.softmax(logits / temperature)

        # split in
        #    usual output probabilities
        #    probabilities made softer with temperature
        y_pred, y_pred_soft = y_pred[:, :self.classes], y_pred[:, self.classes:]

        return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

    def create_model(self):
        from helper import top_3_categorical_acc, precision, recall, f1_score

        inputs = Input(shape=dim, name='input')
        assert self.net_type  in ['mobnet',  'densenet', 'resnet','small_mobnet'], 'Please specify the correct name of architecture from the following : {}'.format(
            ['mobnet', 'densenet', 'resnet','small_mobnet'])
        if self.net_type=='mobnet':
            model_mobilenet = MobileNet(input_shape=dim, alpha=1, depth_multiplier=1, dropout=self.dropout_global,
                                        include_top=False, weights=self.w, input_tensor=None)
            if self.trainable is not None:
                qua_layers=len(model_mobilenet.layers)
                for i in model_mobilenet.layers[qua_layers-self.trainable:]:
                    i.trainable=True

            x = model_mobilenet(inputs)
            x=SeparableConv2D(filters=128,kernel_size=(7,7),activation='relu',padding='same')(x)
            x=BatchNormalization()(x)
            x = Dense(1024, activation='relu',kernel_regularizer=l2(self.regulizer))(x)
            x = Dropout(self.dropout)(x)

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
            x = Dense(256, activation='relu', kernel_regularizer=l2(0.1))(x)
        elif self.net_type=='small_mobnet':

            model_mobilenet_small = MobileNet(input_shape=dim, alpha=0.25, depth_multiplier=1, dropout=self.dropout_global,
                                        include_top=False, weights=self.w, input_tensor=None)
            x = model_mobilenet_small(inputs)

            x = GlobalAveragePooling2D()(x)
            x= Dropout(self.dropout)(x)

        if self.classes==2:
            z = Dense(self.classes)(x)
            z= Activation('sigmoid')(z)
        else:

            z = Dense(self.classes)(x)
            z = Activation('softmax')(z)

        model = Model(inputs=inputs, outputs=z)

        adam = Adam(lr=self.eta, amsgrad=self.amsgrad)
        if self.loss=='custom':
            self.loss=self.loss_regularized
        if not self.train_mode and not self.distillation_mode:
           model.compile(optimizer=adam, loss=self.loss, metrics=['accuracy', top_3_categorical_acc,

                                                               precision, recall, f1_score])
        elif self.distillation_mode:
            model.layers.pop()

            # usual probabilities
            logits = model.layers[-1].output
            probabilities = Activation('softmax')(logits)

            # softed probabilities
            logits_T = Lambda(lambda x: x / self.temperature)(logits)
            probabilities_T = Activation('softmax')(logits_T)

            output = K.concatenate([probabilities, probabilities_T])
            model = Model(model.input, output)
            model.compile(
                optimizer=adam,
                loss=lambda y_true, y_pred: self.knowledge_distillation_loss(y_true, y_pred,
                                                                             self.lambda_const, self.temperature),
                metrics=[self.accuracy]
            )
        else:
            model.compile(optimizer=adam, loss=self.loss, metrics=['accuracy'])

        print(model.summary())

        return model, adam
