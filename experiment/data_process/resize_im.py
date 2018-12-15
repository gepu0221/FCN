import cv2
import os
import numpy as np
import glob
import pdb

root_path = 's8_video_part_3000'
save_path = 's8_video_part_3000_'
file_list = []
glob_file = os.path.join(root_path, '*.bmp')
file_list.extend(glob.glob(glob_file))

if not os.path.exists(save_path):
    os.makedirs(save_path)

for f in file_list:
    print(f)
    fn = os.path.splitext(f.split("/")[-1])[0]
    im = cv2.imread(f)
    im_sz = im.shape
    
    im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_CUBIC)
    save_path_ = os.path.join(save_path, '%s.bmp' % fn)
    cv2.imwrite(save_path_, im)


