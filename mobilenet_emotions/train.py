import datetime
import os
import keras
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from settings import *
from net import *
from keras.callbacks import CSVLogger

class Train:
    def __init__(self):
        """
        initialize all necessary params

        """
        self.train_batch = train_batches
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.train_generator = self.datagen.flow_from_directory(directory=path_to_data,
                                                                target_size=(dim[0],dim[1]), color_mode='rgb',
                                                                batch_size=self.train_batch,
                                                                class_mode='categorical',
                                                                shuffle=True,
                                                                seed=random_state)

        self.classes = self.train_generator.class_indices
        self.now = datetime.datetime.now()
        self.date =  str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + "_" + str(
            self.now.hour) + '-' + str(self.now.minute)
        self.path_to_hist = os.path.join(user, self.date+".csv")
        self.path_to_model = os.path.join(user, 'mobilenet_emotions/models')

        self.epochs = num_epochs
        self.img_size = dim[0]
        self.make_modelsdir()
        self.make_checkpoints()
        self.classes = dict((v, k) for k, v in self.classes.items())


    def make_modelsdir(self):

        if not os.path.exists(self.path_to_model):
            os.mkdir(self.path_to_model)
        self.path_to_model = os.path.join(self.path_to_model, self.date+'.h5')

    def make_checkpoints(self):
        self.path_to_cheks = os.path.join(user, 'mobilenet_emotions/checkpoints')
        if not os.path.exists(self.path_to_cheks):
            os.mkdir(self.path_to_cheks)
        self.path_to_cheks=os.path.join(self.path_to_cheks,self.date+'.h5')
    def fit_nn(self):

        """
        fitting model

        """

        dMN = DeepMN(classes=self.classes)
        model = dMN.create_model()


        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

        callback = keras.callbacks.ModelCheckpoint(filepath= self.path_to_cheks, monitor='acc', verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False, mode='max', period=1)
        csv_logger = CSVLogger(self.path_to_hist, append=True, separator=';')
        callback_List = [callback,csv_logger]

        model.fit_generator(generator=self.train_generator, callbacks=callback_List, epochs=self.epochs,
                                          verbose=1,
                                          shuffle=True, steps_per_epoch=self.train_batch, initial_epoch=0,
                                          use_multiprocessing=False,
                                          workers=12)

        keras.models.save_model(model,self.path_to_model)