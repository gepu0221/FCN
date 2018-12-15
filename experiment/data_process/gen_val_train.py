import cv2
import numpy
import os
import pdb
import glob

root_dir = 'data/inst_da/rect_inst_da_mask1214'
#save_path = 's8_part_video_gtFine3_inter6_1206'
val_rate = 0.1

#if not os.path.exists(save_path):
#    os.makedirs(save_path)

train_path = os.path.join(root_dir, 'train')
val_path = os.path.join(root_dir, 'val')
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)

glob_file = os.path.join(root_dir, '*.bmp')
file_list = []
file_list.extend(glob.glob(glob_file))

len_ = len(file_list)
val_num = int(len_ * val_rate)

for i in range(0, val_num):
    f = file_list[i]
    im = cv2.imread(f)
    fn =  os.path.splitext(f.split("/")[-1])[0]
    path_ = os.path.join(val_path, fn+'.bmp')
    cv2.imwrite(path_, im)

for i in range(val_num, len_):
    f = file_list[i]
    im = cv2.imread(f)
    fn =  os.path.splitext(f.split("/")[-1])[0]
    path_ = os.path.join(train_path, fn+'.bmp')
    cv2.imwrite(path_, im)
