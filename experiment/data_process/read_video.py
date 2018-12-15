#transform video data to images.
import cv2
import os
import numpy as np

#video name
v_name = 's9.mp4'
#save_path
save_path = 's8_video'

if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(v_name)

count = 0
while True:
    #get a frame.
    ret, frame = cap.read()
    fn = os.path.join(save_path, 's8img%05d.bmp' % count)
    count += 1

    cv2.imwrite(fn, frame)



