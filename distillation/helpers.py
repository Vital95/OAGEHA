import keras.preprocessing.image  as preprocess_image
import pickle
import numpy as np
from keras.models import Model, load_model
import pandas as pd
from models import *
from sklearn.model_selection import train_test_split
import os
from multiprocessing.dummy import Pool


def load_data(path_dataframe, abs_path, width=28, height=28, workers=12,step=100):
    def load(x):
        images = []
        target_size = x[1:]
        pathes = x[0]
        for i in pathes:
            image = preprocess_image.img_to_array(preprocess_image.load_img(i, color_mode='rgb',
                                                                            target_size=target_size),
                                                  data_format='channels_last', dtype=np.float32) / 255
            images.append(image)
        return images

    df = pd.read_csv(path_dataframe)
    df['path'] = df['path'].apply(lambda x: os.path.join(abs_path, x))
    X_train, X_test = train_test_split(df, test_size=0.15, random_state=1)
    training_labels, testing_labels = X_train['label'].values, X_test['label'].values
    tup_train = [(X_train['path'].values[i:i + step], height, width) for i in
                 range(0, len(X_train['path'].values), step)]
    tup_test = [(X_test['path'].values[i:i + step], height, width) for i in range(0, len(X_test['path'].values), step)]
    with Pool(workers) as p:
        training_images = p.map(load, tup_train)
        testing_images = p.map(load, tup_test)
    training_images = np.array([z for i in training_images for z in i])
    testing_images = np.array([z for i in testing_images for z in i])
    nb_classes = df['label'].nunique()

    return (training_images, training_labels), (testing_images, testing_labels), nb_classes


def load_teacher_model(model_path):
    model = load_model(model_path, compile=False)
    # remove softmax
    model.layers.pop()
    model = Model(model.input, model.layers[-1].output)
    # now model outputs logits
    return model


def myGenerator(dataset, batch_size):
    while 1:
        n = len(dataset)
        steps = int(np.ceil(n / batch_size))
        for i in range(steps):
            yield i, dataset[i * batch_size:(i + 1) * batch_size]


def save_logits(training_data, model_path, outpaths=('train_logits.npy', 'test_logits.npy'), batch_size=100):
    model = load_teacher_model(model_path)
    (x_train, y_train), (x_test, y_test), nb_classes = training_data

    train_logits = []
    train_generator = myGenerator(x_train, batch_size)
    n = len(x_train)
    total_steps = int(np.ceil(n / batch_size))
    print("Processing train data logits in %d steps by batches of size %d" % (total_steps, batch_size))
    for num_batch, x_batch in train_generator:
        print(len(x_batch))
        print("processing batch %d..." % num_batch)
        batch_logits = model.predict_on_batch(x_batch)

        for i in range(len(batch_logits)):

            train_logits.append(batch_logits[i])

        if num_batch >= total_steps - 1:
            break

    np.save(outpaths[0], train_logits)
    print('Train logits saved to %s' % outpaths[0])

    test_logits = []
    test_generator = myGenerator(x_test, batch_size)
    n = len(x_test)
    total_steps = int(np.ceil(n / batch_size))
    print("Processing test data logits in %d steps by batches of size %d" % (total_steps, batch_size))
    for num_batch, x_batch in test_generator:
        print("processing batch %d..." % num_batch)

        batch_logits = model.predict_on_batch(x_batch)

        for i in range(len(batch_logits)):
            test_logits.append(batch_logits[i])

        if num_batch >= total_steps - 1:
            break

    np.save(outpaths[1], test_logits)
    print('Test logits saved to %s' % outpaths[1])

def make_dir():
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')