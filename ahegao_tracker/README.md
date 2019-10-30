https://drive.google.com/drive/folders/1dxRYvkwInnIKQWj7oyRTyqYRPwA1TDOD?usp=sharing - load the model for emotions from this link
https://drive.google.com/open?id=1Wj57j0lJk1x1_8zRBpbmXMW2mCm0FTJb - load model for yolo object detection (not necessary)

put both models into the directory and run the command : <pre>python run.py</pre> 
in order to switch from tensorflowmodel back to yolo run : <pre>python run.pu --type_model=''</pre>
for tensorflow (default) model it's recommended to use threshold_conf of 0.5, the one can easily change it by running the following command : <pre>python run.py --threshold_conf=0.5</pre>

added new logic for face rotation, described here : https://medium.com/@urumipainblackreaper/precise-face-alignment-with-opencv-dlib-e6c8acead262
