import numpy as np
img_size = 128
dim = (img_size, img_size, 3)
random_state = 12
seed = 16
train_logits_path='train_logits.npy'
dev_logits_path='dev_logits.npy'
mode='rgb'
user = '/home/hikkav/AhegaoProject'
path_to_train = '/home/hikkav/oahega_train'
path_to_valid = '/home/hikkav/oahega_dev'
img_path = '/home/hikkav/Загрузки/archetypal-female-_3249633c.jpg'
classweights=True
space_params_fit_gen = [
    (0.1,0.6), #dumpparam
    (30, 60),  # epochs
    np.arange(32,96,8),  # batch_train
    np.arange(8,40,8),  # batch_valid
    (1e-4, 0.8),  # dropout
    (0.1,0.8),  # dropout_global
    (1e-5, 1e-2), # eta
    ['mobnet'], #repsentation
    [None,5,10,15,20], #trainable layers
    (1e-3,1), #l2
    (1e-3,1) #custom loss beta
]
space_params_fit_distill = [
    (0.1,0.6), #dumpparam
    (30, 60),  # epochs
    np.arange(32,96,8),  # batch_train
    np.arange(8,40,8),  # batch_valid
    (1e-4, 0.8),  # dropout
    (0.1,0.8),  # dropout_global
    (1e-5, 1e-2), # eta
    ['small_mobnet'], #repsentation
    [None,5,10,15,20], #trainable layers
    (1e-4,1e-1),#lambda const
    (2,10) #temperature

]
tune_lr=True
custom_loss=True
ncalls = 20
defined_params=[ 0.343842,60,72,32,  0.693252 , 0.457552 ,  0.000843 ,'custom','mobnet',20,0.007,0.1]