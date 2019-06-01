import argparse

from keras.engine.saving import model_from_json, load_model
from math import atan
import math
import cv2
from PIL import Image
from skimage import transform
import numpy as np
import preprocessing as prep
import os
import sys






def parse_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--threshold_nms', type=float, default=0.8,
                        help='nms threshold')
    parser.add_argument('--classes', type=str, default='ahegao.names',
                        help='path to list with classes to predict')
    parser.add_argument('--qua_conf', type=int, default=4,
                        help='path to list with classes to predict')
    args = parser.parse_args()
    return args

def load_models():

    json = open('MobileNet.json', 'r')
    model = json.read()
    json.close()
    model = model_from_json(model)
    model.load_weights('DMNfullmodel.h5')
    model_emotions = load_model('2019-5-16_0-9.h5')
    return model, model_emotions


def get_classes(args):
    class_list = open(args.classes).read().strip().split('\n')
    return class_list
