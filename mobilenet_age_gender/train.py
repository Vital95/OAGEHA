import argparse
import logging
import os
import random
import string

import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam

from settings import *
from data_generator import ImageGenerator
from data_loader import DataManager, split_imdb_data
from net import DeepMN
from utils import mk_dir

logging.basicConfig(level=logging.DEBUG)


class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.10:
            return 0.001
        elif epoch_idx < self.epochs * 0.25:
            return 0.0001
        elif epoch_idx < self.epochs * 0.60:
            return 0.00005
        return 0.00008


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=70,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split

    logging.debug("Loading data...")

    dataset_name = 'imdb'
    data_loader = DataManager(dataset_name, dataset_path=input_path)
    ground_truth_data = data_loader.get_data()
    train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)

    print("Samples: Training - {}, Validation - {}".format(len(train_keys), len(val_keys)))
    input_shape = dim
    images_path = img_folder

    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                     (128,128),
                                     train_keys, val_keys,
                                     path_prefix=images_path,
                                     vertical_flip_probability=0
                                     )

    model = DeepMN(weights='imagenet')()

    opt = Adam(lr=0.001)

    model.compile(
        optimizer=opt,
        loss=["binary_crossentropy",
              "categorical_crossentropy"],
        metrics={'gender': 'accuracy',
                 'age': 'accuracy'},
    )

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir("models")
    with open(os.path.join("models", "MobileNet.json"), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints")

    run_id = "MobileNet - " + str(batch_size) + " " + '' \
        .join(random
              .SystemRandom()
              .choice(string.ascii_uppercase) for _ in
              range(10)
              )
    print(run_id)

    reduce_lr = ReduceLROnPlateau(
        verbose=1, epsilon=0.001, patience=4)

    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs)),
        reduce_lr,
        ModelCheckpoint(
            os.path.join('checkpoints', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto"
        ),
        TensorBoard(log_dir='logs/' + run_id)
    ]

    logging.debug("Running training...")

    hist = model.fit_generator(
        image_generator.flow(mode='train'),
        steps_per_epoch=int(len(train_keys) / batch_size),
        epochs=nb_epochs,
        callbacks=callbacks,
        validation_data=image_generator.flow('val'),
        validation_steps=int(len(val_keys) / batch_size)
    )

    logging.debug("Saving weights...")
    model.save(os.path.join("models", "DMNfullmodel.h5"))
    model.save_weights(os.path.join("Weights", weights_path), overwrite=True)
    pd.DataFrame(hist.history).to_csv(os.path.join(path_to_hist,"hist.csv") ,sep=',')


if __name__ == '__main__':
    main()
