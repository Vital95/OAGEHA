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
from keras import Model
from tqdm import tqdm
import image_preprocessing_ver2 as impv2
import image_preprocessing_ver1 as impv1



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


class Train:
    def __init__(self, train_batches=40, num_epochs=50, valid=25):
        """
        initialize all necessary params

        """
        self.counter =0
        self.list_with_pathes = []
        self.eval_batch = 100
        self.train_batch = train_batches
        self.dev_batch = valid
        self.datagen = ImageDataGenerator(rescale=1. / 255,

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

    def evaluate(self, args=None, model_path=None, path_to_evaluate=None):
        if model_path is None:
            model_path = args.model_path
        if self.model is None:
            self.model = keras.models.load_model(model_path)
        acc = self.model.evaluate_generator(generator=self.dev_generator, steps=ceil(self.dev_generator.samples / 8),
                                            verbose=1)
        print(acc)

    def preprocess_input(self, x):
        x /= 255.0
        x -= 0.5
        x *= 2.0
        return x

    def get_logits(self, args):
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, \
        drop_global, eta, loss, representation, \
        trainable, l2, beta_loss = defined_params
        data_generator = impv1.ImageDataGenerator(
            preprocessing_function=self.preprocess_input
        )

        train_generator = data_generator.flow_from_directory(
            path_to_train,
            target_size=(img_size, img_size),
            batch_size=self.train_batch, shuffle=False
        )

        val_generator = data_generator.flow_from_directory(
            path_to_valid,
            target_size=(img_size, img_size),
            batch_size=self.dev_batch, shuffle=False
        )
        dn = DeepMN(self.classes, dropout=dropout, loss=loss, trainable=trainable, eta=eta,
                       dropout_global=drop_global,
                       net_type=representation, train_mode=True, l2_=l2, beta_loss=beta_loss)
        model=dn.create_model()
        model.load_weights(args.model)
        model.layers.pop()
        model = Model(model.input, model.layers[-1].output)
        self.save_logits(model, train_generator, self.train_batch, train_logits_path)
        self.save_logits(model, val_generator, self.dev_batch, dev_logits_path)

    def save_logits(self, model, generator, batch, path):
        batches = 0
        logits = {}

        for x_batch, _, name_batch in tqdm(generator):

            batch_logits = model.predict_on_batch(x_batch)

            for i, n in enumerate(name_batch):
                logits[n] = batch_logits[i]

            batches += 1
            if batches >= ceil(generator.samples / batch):
                break
        np.save(path, logits)

    def load_logits(self):
        return np.load(train_logits_path), np.load(dev_logits_path)

    def distill_net(self, params, mode='train'):
        self.counter+=1
        id_ = random.randint(random.randint(25, 601), random.randint(602, 888))
        train_logits, val_logits = self.load_logits()
        data_generator = impv2.ImageDataGenerator(
            data_format='channels_last',
            preprocessing_function=self.preprocess_input
        )

        # note: i'm also passing dicts of logits
        train_generator = data_generator.flow_from_directory(
            path_to_train, train_logits,
            target_size=(img_size, img_size),
            batch_size=32
        )

        val_generator = data_generator.flow_from_directory(
            path_to_valid, val_logits,
            target_size=(128, 128),
            batch_size=16
        )

        if args.from_file and os.path.exists('best_params_distill.json') and mode == 'train':
            distill_params = load_params()

        elif mode == 'train':
            distill_params = defined_distilled_params
        else:
            distill_params = params

        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, \
        eta, loss, representation, \
        trainable, lambda_const, temperature = distill_params

        if classweights:
            from helper import create_class_weight
            classweight = create_class_weight(self.filenames, dump_param)
            print('Created weights for imbalanced data')

        else:
            classweight = None

        print(distill_params)

        dn = DeepMN(self.classes, dropout=dropout, loss=loss, trainable=trainable, eta=eta, dropout_global=drop_global,
                    net_type=representation, train_mode=True, distillation=True,
                    lambda_const=lambda_const, temperature=temperature)
        self.init_generators()
        self.model, _ = dn.create_model()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, )
        checkpoint = keras.callbacks.ModelCheckpoint(self.path_to_cheks, monitor='val_acc', verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=False, mode='max', period=1)
        path_to_hist = self.path_to_hist.split('.csv')[0] + str(self.counter) + '_distilled.csv'
        self.list_with_pathes.append(path_to_hist.split('/')[len(path_to_hist.split('/')) - 1])
        csv_logger = keras.callbacks.CSVLogger(path_to_hist)
        callbacks = [checkpoint, reduce_lr, csv_logger]

        hist = self.model.fit_generator(callbacks=callbacks, generator=strain_generator,
                                        validation_data=val_generator,
                                        validation_steps=ceil(val_generator.samples / self.dev_batch),
                                        steps_per_epoch=ceil(train_generator.samples / self.train_batch),
                                        epochs=self.epochs, class_weight=classweight,
                                        shuffle=True, workers=10, verbose=1)

        if mode == 'train':
            self.model.save(self.path_to_model.split('.')[0] + '_distilled.h5')
            self.model.save_weights(self.path_to_model.split('.')[0] + '_weights_distilled.hdf5')
        else:
            df = pd.read_csv(path_to_hist, sep=',')
            df['train_batch'] = [self.train_batch for i in range(self.epochs)]
            df['valid_batch'] = [self.dev_batch for i in range(self.epochs)]
            df['dropout'] = [dropout for i in range(self.epochs)]
            df['dropout_global'] = [drop_global for i in range(self.epochs)]
            df['eta'] = [eta for i in range(self.epochs)]
            df['loss'] = ['categorical_crossentropy' for i in range(self.epochs)]
            df['dump_param'] = [dump_param for i in range(self.epochs)]
            df['trainable_layers'] = [trainable for i in range(self.epochs)]
            df['experiment_id'] = [id_ for i in range(self.epochs)]
            df['temperature'] = [temperature for i in range(self.epochs)]
            df['lambda_const'] = [lambda_const for i in range(self.epochs)]
            df.to_csv(path_to_hist, index=False, sep=',')
            return -hist.history['val_acc'][len(hist.history['val_acc']) - 1]

    def train_net(self, args):
        if args.from_file and os.path.exists('best_params.json'):
            params = load_params()

        else:
            params = defined_params
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, loss, representation, trainable, l2, beta_loss = params
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
        self.model.save_weights(self.path_to_model.split('.')[0] + 'weights.hdf5')

    def define_params_nn(self, params):
        """
        define best params of model with generator

        """

        self.counter += 1
        id_ = random.randint(random.randint(25, 601), random.randint(602, 888))
        dump_param, self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, representation, trainable, l2, beta_loss = params
        print(
            'Iteration {} ; dump_param : {} , epochs : {} , train batch : {}, valid batch : {} , dropout : {} , '
            'dropout_global : {} , '
            'eta : {} , representation : {}, trainable_layers : {}; l2 : {}, beta_loss : {}'

                .format(self.counter, dump_param,
                        self.epochs, self.train_batch, self.dev_batch, dropout, drop_global, eta, representation,
                        trainable, l2, beta_loss
                        ))

        self.init_generators()
        path_to_hist = self.path_to_hist.split('.csv')[0] + str(self.counter) + '.csv'
        self.list_with_pathes.append(path_to_hist.split('/')[len(path_to_hist.split('/')) - 1])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, )
        report = classification_rep(self.pred_datagen, self.dev_generator.samples, list(self.classes.keys()))
        csv_logger = keras.callbacks.CSVLogger(path_to_hist)

        callbacks = [csv_logger, report, reduce_lr]
        if mode == 'grayscale':
            weights = None
        else:
            weights = 'imagenet'
        if not tune_lr:
            eta = 0.001
        if custom_loss:
            loss = 'custom'
        else:
            loss = 'categorical_crossentropy'
        dn = DeepMN(self.classes, dropout=dropout, loss=loss, trainable=trainable,
                    weights=weights, eta=eta,
                    dropout_global=drop_global, net_type=representation, l2_=l2, beta_loss=beta_loss)
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

        df['train_batch'] = [self.train_batch for i in range(self.epochs)]
        df['valid_batch'] = [self.dev_batch for i in range(self.epochs)]
        df['dropout'] = [dropout for i in range(self.epochs)]
        df['dropout_global'] = [drop_global for i in range(self.epochs)]
        df['eta'] = [eta for i in range(self.epochs)]
        df['loss'] = [loss for i in range(self.epochs)]
        df['dump_param'] = [dump_param for i in range(self.epochs)]
        df['trainable_layers'] = [trainable for i in range(self.epochs)]
        df['experiment_id'] = [id_ for i in range(self.epochs)]
        df['l2'] = [l2 for i in range(self.epochs)]
        df['beta_loss'] = [beta_loss for i in range(self.epochs)]
        df.to_csv(path_to_hist, index=False, sep=',')
        return -hist.history['val_acc'][len(hist.history['val_acc']) - 1]

    def run_minimize(self, args):
        if args.distill:
            space = space_params_fit_distill
        else:
            space = space_params_fit_gen
        params = forest_minimize(self.define_params_nn, dimensions=space, n_calls=ncalls,
                                 verbose=True,
                                 random_state=seed)

        form_dataframe(self.hist_dir, self.list_with_pathes)
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
        if not mode == 'grayscale':
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            np_image = np.array(np_image).astype('float32') / 255
            np_image = transform.resize(np_image, (dim[0], dim[1], 3))

        else:

            np_image = np.array(np_image).astype('float32') / 255
            np_image = transform.resize(np_image, (dim[0], dim[1], 1))

        np_image = np.expand_dims(np_image, axis=0)
        tmp = self.model.predict(np_image)
        prediction = np.argmax(tmp, axis=1)
        pred = self.classes[prediction[0]]
        plot_single_pic(path_to_image, pred)
