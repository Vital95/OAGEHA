import argparse
from keras.engine.saving import model_from_json, load_model
from math import atan
import math
import cv2
from PIL import Image, ImageStat
from skimage import transform
import numpy as np
import preprocessing as prep
import os
import sys
from skopt import forest_minimize
import tensorflow.compat.v1 as tf




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_model',type=str,default='tensorflow',help='type of model to use for object detection')
    parser.add_argument('--model-cfg', type=str, default='yolov2-tiny-ahegao.cfg',
                        help='path to config file')
    parser.add_argument('--model-weights', type=str,
                        default='best.weights',
                        help='path to weights of model')

    parser.add_argument('--video', type=str, default='',
                        help='path to video file')
    parser.add_argument('--src', type=int, default=0,
                        help='source of the camera')

    parser.add_argument('--skip', type=int, default=0,
                        help='how many frames to skip')
    parser.add_argument('--threshold_conf', type=float, default=0.8,
                        help='confidence threshold')
    parser.add_argument('--threshold_nms', type=float, default=0.9,
                        help='nms threshold')
    parser.add_argument('--classes', type=str, default='ahegao.names',
                        help='path to list with classes to predict')
    parser.add_argument('--qua_conf', type=int, default=4,
                        help='number of frames to skip to find average predictions')
    parser.add_argument('--age_constant', type=int, default=0.8,
                        help='constant to scale age')
    parser.add_argument('--brightness_threshold', type=int, default=70,
                        help='if brightness threshold is < determined -> more brightness wil be added to input data')

    args = parser.parse_args()
    return args

def load_models():

    json = open('MobileNet.json', 'r')
    model = json.read()
    json.close()
    model = model_from_json(model)
    model.load_weights('DMNfullmodel.h5')
    model_emotions = load_model('distilled_model.h5',compile=False)
    return model, model_emotions


def get_classes(args):
    class_list = open(args.classes).read().strip().split('\n')
    return class_list
