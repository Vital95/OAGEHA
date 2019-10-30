import argparse
import json
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.metrics import top_k_categorical_accuracy
from net import *
from scipy import ndimage
from settings import *
from train import Train

ap = argparse.ArgumentParser()
net = Train()


def parse_args():
    subparsers = ap.add_subparsers()
    fit_parser = subparsers.add_parser('train_net', help='fit mobilenet with the best params defined in settings or '
                                                         'with params from file')

    fit_parser.add_argument('-f', dest='from_file', required=False, type=bool,
                            help='True if you want to load params from file, else -> '
                                 'False')
    fit_parser.set_defaults(func=net.train_net)

    predict_on_single_parser = subparsers.add_parser('predict_on_single_image',
                                                     help='get a prediction for a single img')
    predict_on_single_parser.add_argument('-i', dest='image_path', required=True, type=str,
                                          help='path to pic for making '
                                               ' a prediction ')
    predict_on_single_parser.add_argument('-m', dest='model_path',
                                          help='path to trained model', required=True, type=str,
                                          )
    predict_on_single_parser.set_defaults(func=net.predict)
    evaluate = subparsers.add_parser('evaluate',
                                                     help='evaluate model on unseen test data')
    evaluate.add_argument('-m', dest='model_path',
                                          help='path to trained model', required=True, type=str,
                                          )
    evaluate.set_defaults(func=net.evaluate)
    define_params_minimize = subparsers.add_parser('define_params', help='define params with forest minimize')
    define_params_minimize.set_defaults(func=net.run_minimize)

    return ap.parse_args()


def plot_single_pic(img, label):
    """
    shows a single pic with predicted class
    :param img: the converted img
    :param label: it's predicted class

    """
    img = np.array(ndimage.imread(img, flatten=False))
    ax = plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(img)
    plt.text(0.5, -0.1, label, horizontalalignment='center', verticalalignment='center',
             fontsize=15, transform=ax.transAxes)
    plt.show()


def precision(y_true, y_pred):
    """Precision metric.	
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.	
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.	
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    """Computes the F1 Score
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())


def top_3_categorical_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_pred=y_pred, y_true=y_true, k=3)

def load_params():
    with open('best_params.json', 'r') as bp:
        param = json.loads(bp.read())

    return param['x']


def create_class_weight(labels_dict, mu=1):
    total = np.sum(list(labels_dict.values())) / len(labels_dict)
    keys = list(labels_dict.keys())

    class_weight = dict()

    for key in keys:
        tmp = labels_dict[key]
        score = float(tmp) / total
        class_weight[key] = 1 / (score*mu) if score < 1.0 else 1.0
    print(class_weight)
    return class_weight


def write_best_params(params):
    with open('best_params.json', 'w') as f:
        json.dump(params, f)
