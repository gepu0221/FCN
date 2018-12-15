import numpy as np
import cv2
import os
import pdb

s_num = 10004
inter_num = 2
anno_root = 'data/s8_video_label2_ori3_Occ1204'
im_root = 's8_video_part'
save_root = 'data/s8_part_video_gtFine3_inter6_1206'

if not os.path.exists(save_root):
    os.makedirs(save_root)

while True:
    fn = '%simg%5d.bmp' % ('s8', s_num)
    anno_path = os.path.join(anno_root, fn)
    #pdb.set_trace()
    
    if os.path.exists(anno_path):
        f = True
        for dt in range(-inter_num + 1, 1):
            t = s_num + dt*6
            fn_ = '%simg%05d.bmp' % ('s8', t)
            im_path = os.path.join(im_root, fn_)
          
            #pdb.set_trace()
            if not os.path.exists(im_path):
                #pdb.set_trace()
                f = False
                break
        
        if f:
            path = os.path.join(save_root, fn)
            im = cv2.imread(anno_path)
            cv2.imwrite(path, im)
    else:
        break

    s_num += 1
            
