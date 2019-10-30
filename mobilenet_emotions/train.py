import datetime
import os
import random
from math import ceil
import cv2
import keras
import pandas as pd
from skimage import transform
from custom_callbacks import AdditionalMetrics
from helper import *
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from skopt import forest_minimize
import  tensorflow as tf

class Train:
    def __init__(self, train_batches=40, valid=25, eval_batch=100, num_epochs=50, ):
        """
        initialize all necessary params

        """
        self.list_with_pathes = []
        self.eval_batch = eval_batch
        self.train_batch = train_batches
        self.dev_batch = valid
        self.datagen = ImageDataGenerator(rescale=1. / 255,
                                          zoom_range=[1.0, 1.5],
                                          horizontal_flip=True,
                                          fill_mode='nearest')
        self.pred_datagen = ImageDataGenerator(rescale=1. / 255
                                               )
        self.counter = 0
        self.make_df()
        self.init_generators()
        self.classes = self.train_generator.class_indices
        self.now = datetime.datetime.now()
        self.date = str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + "_" + str(
            self.now.hour) + '-' + str(self.now.minute)
        self.hist_dir = user + '/visualize'
        self.path_to_hist = os.path.join(self.hist_dir, 'history_emotions') + self.date + ".csv"
        self.path_to_model = os.path.join(user, 'mobilenet_emotions/models')
        self.epochs = num_epochs
        self.img_size = dim[0]
        self.make_modelsdir()
        self.make_checkpoints()
        self.classes = dict((v, k) for k, v in self.classes.items())
        self.model = None

    def make_df(self):
        df = pd.read_csv(path_to_data)
        df['path'] = df['path'].apply(lambda x: abs_path + x)
        X = df['path']
        y = df['label']
        skf = StratifiedShuffleSplit(random_state=seed, n_splits=2, test_size=0.15)
        X_train, X_dev, X_test = None, None, None
        y_train, y_dev, y_test = None, None, None
        for train_index, dev_index in skf.split(X, y):
            X_train, X_dev = X.iloc[train_index], X.iloc[dev_index]
            y_train, y_dev = y.iloc[train_index], y.iloc[dev_index]
        skf = StratifiedShuffleSplit(random_state=seed, n_splits=2, test_size=0.10)
        X_train2, y_train2 = None, None
        for train_index2, test_index in skf.split(X_train, y_train):
            X_train2, X_test = X_train.iloc[train_index2], X_train.iloc[test_index]
            y_train2, y_test = y_train.iloc[train_index2], y_train.iloc[test_index]
        X_train, y_train = X_train2, y_train2
        self.valid_df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.valid_df['path'] = X_dev
        self.valid_df['label'] = y_dev
        self.train_df['path'] = X_train
        self.train_df['label'] = y_train
        self.test_df['path'] = X_test
        self.test_df['label'] = y_test

    def init_generators(self):
        self.dev_generator = self.pred_datagen.flow_from_dataframe(dataframe=self.valid_df,
                                                                   target_size=(dim[0], dim[1]), color_mode='rgb',
                                                                   batch_size=self.dev_batch,
                                                                   x_col='path',
                                                                   y_col='label',
                                                                   class_mode='categorical',
                                                                   shuffle=False,
                                                                   seed=random_state,
                                                                   )
        self.test_generator = self.pred_datagen.flow_from_dataframe(dataframe=self.test_df,
                                                                    target_size=(dim[0], dim[1]), color_mode='rgb',
                                                                    batch_size=self.eval_batch,
                                                                    x_col='path',
                                                                    y_col='label',
                                                                    class_mode='categorical',
                                                                    shuffle=False,
                                                                    seed=random_state,
                                                                    )
        self.train_generator = self.datagen.flow_from_dataframe(dataframe=self.train_df,
                                                                target_size=(dim[0], dim[1]), color_mode='rgb',
                                                                batch_size=self.train_batch,
                                                                x_col='path',
                                                                y_col='label',
                                                                class_mode='categorical',
                                                                shuffle=False,
                                                                seed=random_state,
                                                                )

        self.filenames = dict(self.train_df.label.value_counts())

        self.classes = self.train_generator.class_indices
        self.filenames = dict([(self.classes[k], v) for k, v in self.filenames.items()])
        print(self.filenames)
        self.now = datetime.datetime.now()
        self.date = str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + "_" + str(
            self.now.hour) + '-' + str(self.now.minute)

    def make_modelsdir(self):

        if not os.path.exists(self.path_to_model):
            os.mkdir(self.path_to_model)
        self.path_to_model = os.path.join(self.path_to_model, self.date + '.h5')

    def make_checkpoints(self):
        self.path_to_cheks = os.path.join(user, 'mobilenet_emotions/checkpoints')
        if not os.path.exists(self.path_to_cheks):
            os.mkdir(self.path_to_cheks)
        self.path_to_cheks = os.path.join(self.path_to_cheks, self.date + '.h5')

    def evaluate(self, args=None):
        from helper import top_3_categorical_acc, recall, f1_score, precision
        if self.model is None:
            model_path = args.model_path
            try:
                self.model = keras.models.load_model(model_path)
            except:
                self.model = keras.models.load_model(model_path,
                                                     custom_objects={'top_3_categorical_acc': top_3_categorical_acc,
                                                                     'precision': precision, 'recall': recall,
                                                                     'f1_score': f1_score})
        acc = self.model.evaluate_generator(generator=self.test_generator,
                                            steps=ceil(self.test_generator.samples / self.eval_batch),
                                            verbose=1)

        metrics = dict([(i[0], i[1]) for i in zip(self.model.metrics_names, acc)])
        print(metrics)

    def train_net(self, args):
        if args.from_file and os.path.exists('best_params.json'):
            params = load_params()

        else:
            params = defined_params
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, loss, representation, trainable, l2, beta_loss = \
            params['x']
        if classweights:
            from helper import create_class_weight
            classweight = create_class_weight(self.filenames, dump_param)
            print('Created weights for imbalanced data')

        else:
            classweight = None
        print(params)

        dn = DeepMN(self.classes, dropout=dropout, loss=loss, trainable=trainable, eta=eta, dropout_global=drop_global,
                    net_type=representation, train_mode=True, l2_=l2, beta_loss=beta_loss)
        self.init_generators()
        self.model, _ = dn.create_model()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, )
        checkpoint = keras.callbacks.ModelCheckpoint(self.path_to_cheks, monitor='val_acc', verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=False, mode='max', period=1)
        callbacks = [checkpoint, reduce_lr]

        self.model.fit_generator(callbacks=callbacks, generator=self.train_generator,
                                 validation_data=self.dev_generator,
                                 validation_steps=ceil(self.dev_generator.samples / self.dev_batch),
                                 steps_per_epoch=ceil(self.train_generator.samples / self.train_batch),
                                 epochs=self.epochs, class_weight=classweight,
                                 shuffle=True, workers=10, verbose=1)
        self.model.save(self.path_to_model)

    def define_params_nn(self, params):
        """
        define best params of model with generator

        """
        keras.backend.clear_session()
        id_ = random.randint(random.randint(25, 601), random.randint(602, 888))
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, representation, trainable, l2, layer_params = params
        print(
            'dump_param : {} , epochs : {} , train batch : {}, valid batch : {} , dropout : {} , '
            'dropout_global : {} , '
            'eta : {} , representation : {}, frozen layers : {}; l2 : {}, dense_layers : {}'

                .format(dump_param,
                        self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, representation,
                        trainable, l2, layer_params
                        ))

        self.init_generators()
        dict_logs = {}
        if not tune_lr:
            eta = 0.001

        dict_logs['train_batch'] = self.train_batch
        dict_logs['valid_batch'] = self.dev_batch
        dict_logs['dropout'] = dropout
        dict_logs['dropout_global'] = drop_global
        dict_logs['eta'] = eta
        dict_logs['layers_toadd'] = layer_params
        dict_logs['dump_param'] = dump_param
        dict_logs['trainable_layers'] = trainable
        dict_logs['experiment_id'] = id_
        dict_logs['l2'] = l2

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, )
        additional_metrics = AdditionalMetrics(self.pred_datagen, self.dev_generator.samples, list(self.classes.keys()),
                                               self.valid_df, dict_logs)
        if self.counter == 0:
            csv_logger = keras.callbacks.CSVLogger(self.path_to_hist, append=False)
        else:
            csv_logger = keras.callbacks.CSVLogger(self.path_to_hist, append=True)
        self.counter += 1
        logdir = 'tensorboard_logs/scalars/model{}'.format(id_)
        tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()
        callbacks = [additional_metrics, reduce_lr, csv_logger, tensorboard]

        dn = DeepMN(self.classes, dropout=dropout, trainable=trainable,
                    weights='imagenet', eta=eta,
                    dropout_global=drop_global, net_type=representation, l2_=l2, )
        self.model, _ = dn.create_model()

        if classweights:
            from helper import create_class_weight
            classweight = create_class_weight(self.filenames, dump_param)
            print('Created weights for imbalanced data')

        else:
            classweight = None
        hist = self.model.fit_generator(callbacks=callbacks, generator=self.train_generator,
                                        validation_data=self.dev_generator,
                                        validation_steps=ceil(self.dev_generator.samples / self.dev_batch),
                                        steps_per_epoch=ceil(self.train_generator.samples / self.train_batch),
                                        epochs=self.epochs, class_weight=classweight
                                        )

        self.model.save('models/model{0}.h5'.format(hist.history['val_acc'][len(hist.history['val_acc']) - 1]))
        return -hist.history['val_acc'][len(hist.history['val_acc']) - 1]

    def run_minimize(self, args=None):
        from helper import write_best_params

        space = space_params_fit_gen
        params = forest_minimize(self.define_params_nn, dimensions=space, n_calls=ncalls,
                                 verbose=True,
                                 random_state=seed)

        write_best_params(params)
        print('Best params are : {}'.format(params))

    def predict(self, args=None, path_to_image=None, path_to_model=None):
        from helper import plot_single_pic
        if args is not None:
            path_to_model = args.model_path
            path_to_image = args.image_path
        if self.model is None:
            self.model = keras.models.load_model(path_to_model)

        np_image = cv2.imread(path_to_image)

        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (dim[0], dim[1], 1))

        np_image = np.expand_dims(np_image, axis=0)
        tmp = self.model.predict(np_image)
        prediction = np.argmax(tmp, axis=1)
        pred = self.classes[prediction[0]]
        plot_single_pic(path_to_image, pred)
