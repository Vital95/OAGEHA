from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation
from keras import Model, Input
from keras.applications import MobileNet
from keras.regularizers import l2

def build_small_mobnet(width=28, height=28, alpha=0.25, verbose=True):

    dim = (height, width, 3)
    input = Input(shape=dim, name='input')
    model = MobileNet(input_shape=dim, alpha=alpha, depth_multiplier=1,
                      include_top=False, weights='imagenet', input_tensor=None)
    x = model(input)
    x = GlobalAveragePooling2D()(x)

    x = Dense(units=512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    z = Dense(6)(x)
    z = Activation('softmax')(z)
    model = Model(inputs=input, outputs=z)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if verbose == True: print(model.summary())
    return model

