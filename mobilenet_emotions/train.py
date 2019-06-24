import random
from collections import Counter

import numpy as np
import keras
from helper import *
from keras_preprocessing.image import ImageDataGenerator
from skopt import forest_minimize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau
counter = 0


class classification_rep(keras.callbacks.Callback):
    def __init__(self, pred_datagen, samples, names):

        super().__init__()
        self.valid_data = pred_datagen.flow_from_directory(directory=path_to_valid,
                                                           target_size=(dim[0], dim[1]), color_mode=mode,
                                                           batch_size=samples,
                                                           class_mode='categorical',
                                                           shuffle=True,
                                                           seed=random_state,
                                                           )
        self.names = names

    def on_epoch_end(self, epoch, logs=None):
        report = None

        for i in range(len(self.valid_data)):
            x_test_batch, y_test_batch = self.valid_data.__getitem__(i)
            val_predict = (np.asarray(self.model.predict(x_test_batch))).round()
            val_targ = y_test_batch

            report = classification_report(y_true=val_targ, y_pred=val_predict, output_dict=True,
                                           target_names=self.names)

        for i in report.items():
            for z in i[1].items():
                logs[i[0] + '_' + z[0]] = z[1]

                print(str(i[0] + '_' + z[0]) + ' : {}'.format(z[1]))


class auc_per_class(keras.callbacks.Callback):
    def __init__(self):
        self.eval_batch = 100
        self.pred_datagen = ImageDataGenerator(rescale=1. / 255
                                               )
        self.ahegao_gen = self.pred_datagen.flow_from_directory(directory=path_to_ahegao,
                                                                target_size=(dim[0], dim[1]), color_mode=mode,
                                                                batch_size=self.eval_batch,
                                                                class_mode='categorical',
                                                                shuffle=False,
                                                                seed=random_state,
                                                                )
        self.angry_gen = self.pred_datagen.flow_from_directory(directory=path_to_angry,
                                                               target_size=(dim[0], dim[1]), color_mode=mode,
                                                               batch_size=self.eval_batch,
                                                               class_mode='categorical',
                                                               shuffle=False,
                                                               seed=random_state,
                                                               )
        self.happy_gen = self.pred_datagen.flow_from_directory(directory=path_to_happy,
                                                               target_size=(dim[0], dim[1]), color_mode=mode,
                                                               batch_size=self.eval_batch,
                                                               class_mode='categorical',
                                                               shuffle=False,
                                                               seed=random_state,
                                                               )
        self.sad_gen = self.pred_datagen.flow_from_directory(directory=path_to_sad,
                                                             target_size=(dim[0], dim[1]), color_mode=mode,
                                                             batch_size=self.eval_batch,
                                                             class_mode='categorical',
                                                             shuffle=False,
                                                             seed=random_state,
                                                             )
        self.neutral_gen = self.pred_datagen.flow_from_directory(directory=path_to_neutral,
                                                                 target_size=(dim[0], dim[1]), color_mode=mode,
                                                                 batch_size=self.eval_batch,
                                                                 class_mode='categorical',
                                                                 shuffle=False,
                                                                 seed=random_state,
                                                                 )
        self.surprise_gen = self.pred_datagen.flow_from_directory(directory=path_to_surprise,
                                                                  target_size=(dim[0], dim[1]), color_mode=mode,
                                                                  batch_size=self.eval_batch,
                                                                  class_mode='categorical',
                                                                  shuffle=False,
                                                                  seed=random_state,
                                                                  )
        self.fear_gen = self.pred_datagen.flow_from_directory(directory=path_to_fear,
                                                              target_size=(dim[0], dim[1]), color_mode=mode,
                                                              batch_size=self.eval_batch,
                                                              class_mode='categorical',
                                                              shuffle=False,
                                                              seed=random_state,
                                                              )
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        ahegao_hist = self.auc(self.ahegao_gen)
        angry_hist = self.auc(self.angry_gen)
        happy_hist = self.auc(self.happy_gen)
        sad_hist = self.auc(self.sad_gen)
        neutral_hist = self.auc(self.neutral_gen)
        surprise_hist = self.auc(self.surprise_gen)
        fear_hist = self.auc(self.fear_gen)
        logs['fear_acc'] = fear_hist
        logs['ahegao_acc'] = ahegao_hist
        logs['happy_acc'] = happy_hist
        logs['sad_acc'] = sad_hist
        logs['angry_acc'] = angry_hist
        logs['neutral_acc'] = neutral_hist
        logs['surprise_acc'] = surprise_hist
        print('Happy auc : {} , Sad auc : {}, Ahegao auc : {}, Neutral auc : {}, Surprise auc : {},'
              'Angry auc : {}, Fear auc : {}'.format(happy_hist, sad_hist, ahegao_hist, neutral_hist, surprise_hist,
                                                     angry_hist
                                                     , fear_hist))

    def auc(self, gen):
        _, y = gen.next()
        pred = self.model.predict_generator(generator=gen, steps=1)
        prediction = np.argmax(pred, axis=1)
        return accuracy_score(y, np.rint(prediction))


class Train:
    def __init__(self, train_batches=40, num_epochs=50, valid=25):
        """
        initialize all necessary params

        """
        self.list_with_pathes = []
        self.eval_batch = 100
        self.train_batch = train_batches
        self.dev_batch = valid
        self.datagen = ImageDataGenerator(rescale=1. / 255,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          vertical_flip=True,
                                          fill_mode='nearest')
        self.pred_datagen = ImageDataGenerator(rescale=1. / 255
                                               )
        self.init_generators()
        self.classes = self.train_generator.class_indices
        self.now = datetime.datetime.now()
        self.date = str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + "_" + str(
            self.now.hour) + '-' + str(self.now.minute)
        self.hist_dir = user + '/visualize'
        self.path_to_hist = os.path.join(self.hist_dir, self.date + ".csv")
        self.path_to_model = os.path.join(user, 'mobilenet_emotions/models')
        self.epochs = num_epochs
        self.img_size = dim[0]
        self.make_modelsdir()
        self.make_checkpoints()
        self.classes = dict((v, k) for k, v in self.classes.items())
        self.model = None

    def init_generators(self):
        self.train_generator = self.datagen.flow_from_directory(directory=path_to_train,
                                                                target_size=(dim[0], dim[1]), color_mode=mode,
                                                                batch_size=self.train_batch,
                                                                class_mode='categorical',
                                                                shuffle=True,
                                                                seed=random_state,
                                                                )
        self.dev_generator = self.pred_datagen.flow_from_directory(directory=path_to_valid,
                                                                   target_size=(dim[0], dim[1]), color_mode=mode,
                                                                   batch_size=self.dev_batch,
                                                                   class_mode='categorical',
                                                                   shuffle=True,
                                                                   seed=random_state,
                                                                   )

        filenames = self.train_generator.filenames
        filenames = Counter([x.split('/')[0] for x in filenames])
        self.classes = self.train_generator.class_indices
        self.filenames = dict([(self.classes[k], v) for k, v in filenames.items()])
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

    def train_net(self, args):
        if args.from_file and os.path.exists('best_params_sequence.json'):
            params = load_params()

        else:
            params = defined_params
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, loss,representation = params
        if classweights:
            from helper import create_class_weight
            classweight = create_class_weight(self.filenames, dump_param)
            print('Created weights for imbalanced data')

        else:
            classweight = None
        dn = DeepMN(self.classes, dropout=dropout, loss=loss, eta=eta, dropout_global=drop_global,net_type=representation)
        self.init_generators()
        self.model, _ = dn.create_model()
        early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=self.epochs // 3, verbose=0,
                                                   mode='max',
                                                   baseline=None,
                                                   restore_best_weights=True)
        checkpoint = keras.callbacks.ModelCheckpoint(self.path_to_cheks, monitor='val_acc', verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=False, mode='max', period=1)
        callbacks = [checkpoint]

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
        global counter
        counter += 1
        id_ = random.randint(random.randint(25, 601), random.randint(602, 888))
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, representation = params
        print(
            'Iteration {} ; dump_param : {} , epochs : {} , train batch : {}, valid batch : {} , dropout : {} , '
            'dropout_global : {} , '
            'eta : {} , representation : {}'

                .format(counter, dump_param,
                        self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, representation
                        ))

        self.init_generators()
        path_to_hist = self.path_to_hist.split('.csv')[0] + str(counter) + '.csv'
        self.list_with_pathes.append(path_to_hist.split('/')[len(path_to_hist.split('/')) - 1])
        auc = auc_per_class()
        reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,)
        report = classification_rep(self.pred_datagen, self.dev_generator.samples, list(self.classes.keys()))
        csv_logger = keras.callbacks.CSVLogger(path_to_hist)
        callbacks = [csv_logger, report,reduce_lr]
        if mode == 'grayscale':
            weights = None
        else:
            weights = 'imagenet'
        dn = DeepMN(self.classes, dropout=dropout, loss='categorical_crossentropy', weights=weights, eta=eta,
                    dropout_global=drop_global, net_type=representation)
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

        df = pd.read_csv(path_to_hist, sep=',')
        for i in list(hist.history.keys()):
            if str(i).split('_')[0] in list(self.classes.keys()):
                df[i] = hist.history[i]
        # df['fear_acc'] = hist.history['fear_acc']
        # df['ahegao_acc'] = hist.history['ahegao_acc']
        # df['happy_acc'] = hist.history['happy_acc']
        # df['sad_acc'] = hist.history['sad_acc']
        # df['angry_acc'] = hist.history['angry_acc']
        # df['neutral_acc'] = hist.history['neutral_acc']
        # df['surprise_acc'] = hist.history['surprise_acc']
        df['train_batch'] = [self.train_batch for i in range(self.epochs)]
        df['valid_batch'] = [self.dev_batch for i in range(self.epochs)]
        df['dropout'] = [dropout for i in range(self.epochs)]
        df['dropout_global'] = [drop_global for i in range(self.epochs)]
        df['eta'] = [eta for i in range(self.epochs)]
        df['loss'] = ['categorical_crossentropy' for i in range(self.epochs)]
        df['dump_param'] = [dump_param for i in range(self.epochs)]
        df['experiment_id'] = [id_ for i in range(self.epochs)]
        df.to_csv(path_to_hist, index=False, sep=',')
        return -hist.history['val_acc'][len(hist.history['val_acc']) - 1]

    def run_minimize(self, args):

        params = forest_minimize(self.define_params_nn, dimensions=space_params_fit_gen, n_calls=ncalls,
                                 verbose=True,
                                 random_state=seed)

        form_dataframe(self.hist_dir, self.list_with_pathes)
        write_best_params(params)
        print('Best params are : {}'.format(params))

    def predict(self, path_to_image, path_to_model, args=None):
        if args is not None:
            path_to_model = args.model_path
            path_to_image = args.image_path
        if self.model is None:
            self.model = keras.models.load_model(path_to_model)
        np_image = cv2.imread(path_to_image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (dim[0], dim[1], 3))
        np_image = np.expand_dims(np_image, axis=0)
        tmp = self.model.predict(np_image)
        prediction = np.argmax(tmp, axis=1)
        pred = self.classes[prediction[0]]
        plot_single_pic(path_to_image, pred)
