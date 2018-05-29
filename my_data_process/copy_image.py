import tensorflow as tf
import numpy as np
import cv2
import os

image_folder='/home/gp/repos/data/train_data/s5/Image'

frame_name_list=[f for f in os.listdir(image_folder) if f.endswith(".bmp")]


for f in frame_name_list:
    im = cv2.imread(f,cv2.IMREAD_COLOR)
    print('frame_name',f)
    im_name_split=f.split('/')
    len_=len(im_name_split)
    frame_name=im_name_split[len_-1]
    new_frame_name=frame_name.split('.')[0]+'c'
    frame_name=new_frame_name+'.bmp'
    print('new_frame_name',frame_name)
    frame_path=os.path.join(image_folder,frame_name)
    cv2.imwrite(frame_path,im)
                          