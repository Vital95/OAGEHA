from train import *
import numpy as np
from skimage import transform
import cv2
net_=Train()
model = keras.models.load_model('/home/hikkav/AhegaoProject/mobilenet_emotions/checkpoints/2019-5-16_0-9.h5')
np_image = cv2.imread('/home/hikkav/AhegaoProject/tmp.jpg')
np_image=cv2.cvtColor(np_image,cv2.COLOR_RGB2GRAY)
np_image = np.array(np_image).astype('float32') / 255
np_image = transform.resize(np_image, (dim[0], dim[1], 3))
np_image = np.expand_dims(np_image, axis=0)
tmp = model.predict(np_image)
print(tmp)
prediction = np.argmax(tmp,axis=1)
print(net_.classes)
pred = net_.classes[prediction[0]]
print(pred)