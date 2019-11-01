# Mute tensorflow debugging information console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import save_model, Model
from keras.utils import np_utils
from keras.layers import Lambda, concatenate, Activation
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
import keras
from keras import backend as K
import argparse
import numpy as np
import datetime
from helpers import load_data, save_logits, make_dir
from models import build_small_mobnet
from sklearn.preprocessing import LabelBinarizer


def train(model, model_label, training_data, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test),  nb_classes = training_data

    # convert class vectors to binary class matrices
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    STAMP = model_label
    print('Training model {}'.format(STAMP))
    logs_path = './logs/{}'.format(STAMP)

    bst_model_path = 'checkpoints/' + STAMP.format(datetime.datetime.now()) + '.h5'
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
    tensor_board = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1)

    hist = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, tensor_board, reduce_lr])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    save_model(model, 'model/'+STAMP.format(hist.history['val_loss'][-1])+'model.h5')

def train_student(model, model_label, training_data, teacher_model_path,
                  logits_paths=('train_logits.npy', 'test_logits.npy'),
                  batch_size=256, epochs=10, temp=5.0, lambda_weight=0.1):
    temperature = temp
    (x_train, y_train), (x_test, y_test),  nb_classes = training_data

    # convert class vectors to binary class matrices
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    # load or calculate logits of trained teacher model
    train_logits_path = logits_paths[0]
    test_logits_path = logits_paths[1]
    if not (os.path.exists(train_logits_path) and os.path.exists(test_logits_path)):
        save_logits(training_data, teacher_model_path,logits_paths)
    train_logits = np.load(train_logits_path)
    test_logits = np.load(test_logits_path)

    # concatenate true labels with teacher's logits
    y_train = np.concatenate((y_train, train_logits), axis=1)
    y_test = np.concatenate((y_test, test_logits), axis=1)

    # remove softmax
    model.layers.pop()
    # usual probabilities
    logits = model.layers[-1].output
    probabilities = Activation('softmax')(logits)

    # softed probabilities
    logits_T = Lambda(lambda x: x / temperature)(logits)
    probabilities_T = Activation('softmax')(logits_T)

    output = concatenate([probabilities, probabilities_T])
    model = Model(model.input, output)
    # now model outputs 26+26 dimensional vectors

    #internal functions
    def knowledge_distillation_loss(y_true, y_pred, lambda_const):
        # split in
        #    onehot hard true targets
        #    logits from teacher model
        y_true, logits = y_true[:, :nb_classes], y_true[:, nb_classes:]

        # convert logits to soft targets
        y_soft = K.softmax(logits / temperature)

        # split in
        #    usual output probabilities
        #    probabilities made softer with temperature
        y_pred, y_pred_soft = y_pred[:, :nb_classes], y_pred[:, nb_classes:]

        return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

    # For testing use usual output probabilities (without temperature)
    def acc(y_true, y_pred):
        y_true = y_true[:, :nb_classes]
        y_pred = y_pred[:, :nb_classes]
        return categorical_accuracy(y_true, y_pred)

    def categorical_crossentropy(y_true, y_pred):
        y_true = y_true[:, :nb_classes]
        y_pred = y_pred[:, :nb_classes]
        return logloss(y_true, y_pred)

    # logloss with only soft probabilities and targets
    def soft_logloss(y_true, y_pred):
        logits = y_true[:, nb_classes:]
        y_soft = K.softmax(logits / temperature)
        y_pred_soft = y_pred[:, nb_classes:]
        return logloss(y_soft, y_pred_soft)

    lambda_const = lambda_weight

    model.compile(
        #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
        optimizer='adadelta',
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
        metrics=[acc, categorical_crossentropy, soft_logloss]
    )

    STAMP = model_label
    print('Training model {}'.format(STAMP))
    logs_path = './logs/{}'.format(STAMP)

    bst_model_path = 'checkpoints/' + STAMP.format(datetime.datetime.now()) + '.h5'
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True,
                                                       verbose=1)
    tensor_board = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True,
                                               write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1)

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, reduce_lr, tensor_board])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    save_model(model, 'model/'+STAMP.format(history.history['categorical_crossentropy'][-1])+'model.h5')


if __name__ == '__main__':
    make_dir()
    parser = argparse.ArgumentParser(usage='A training program for classifying the OAHEGA dataset')
    parser.add_argument('-f', '--file', type=str, help='Path to csv file data',
                        required=True)
    parser.add_argument('-a', '--abs_path', type=str, help='Absolute path to data',
                        required=True)
    parser.add_argument('-m', '--model', type=str, help='model to be trained (student,distill).'
                                                        ' If student is selected than path to pretrained teacher must be specified in --teacher parameter',
                        required=True)
    parser.add_argument('-t', '--teacher', type=str, help='path to .h5 file with weight of pretrained teacher model'
                                                         ,
                        default='../mobilenet_emotions/models/model0.8267208413001912.h5')

    parser.add_argument('--width', type=int, default=128, help='Width of the images')
    parser.add_argument('--height', type=int, default=128, help='Height of the images')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs to train on')
    args = parser.parse_args()

    if not os.path.exists('model/'):
        os.mkdir('model/')

    training_data = load_data(args.file, width=args.width, height=args.height, abs_path=args.abs_path)


    if args.model=='student':
        model =build_small_mobnet(width=args.width, height=args.height, verbose=True)
        temp = 2.0
        lamb = 0.5
        label = 'student_model_distill_{}'

        train_student(model,label,training_data, teacher_model_path=args.teacher,
                      epochs=args.epochs, temp=temp, lambda_weight=lamb)
    else:
        label = 'student_model_orig_{}'
        model = build_small_mobnet(width=args.width, height=args.height, verbose=True)
        train(model, label, training_data, epochs=args.epochs)

