import numpy as np
img_size = 128
dim = (img_size, img_size, 1)
random_state = 9
seed = 51
mode='grayscale'
user = '/home/hikkav/AhegaoProject'
path_to_train = '/home/hikkav/AhegaoProject/ferdataset_train'
path_to_valid = '/home/hikkav/AhegaoProject/ferdataset_valid'
path_to_angry='/home/hikkav/AhegaoProject/Angry'
path_to_ahegao='/home/hikkav/AhegaoProject/Ahegao'
path_to_sad='/home/hikkav/AhegaoProject/Sad'
path_to_surprise='/home/hikkav/AhegaoProject/Surprise'
path_to_neutral='/home/hikkav/AhegaoProject/Neutral'
path_to_fear='/home/hikkav/AhegaoProject/Fear'
path_to_happy='/home/hikkav/AhegaoProject/Happy'
img_path = '/home/hikkav/Загрузки/archetypal-female-_3249633c.jpg'
classweights=True
space_params_fit_gen = [
    (0.1,0.6), #dumpparam
    (90, 150),  # epochs
    np.arange(32,96,8),  # batch_train
    np.arange(8,40,8),  # batch_valid
    (1e-2, 0.8),  # dropout
    (1e-5, 1e-3),  # dropout_global
    (1e-4, 1e-3), # eta
    ('resnet','mobnet') #repsentation
]

ncalls = 40
defined_params=[]