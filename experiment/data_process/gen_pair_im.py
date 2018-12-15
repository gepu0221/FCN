import numpy as np
import cv2
import os
import pdb

s_num = 10004
inter_num = 2
anno_root = 'data/s8_video_label2_ori3_Occ1204'
im_root = 's8_video_part'
save_root = 'data/s8_part_video_gtFine3_inter40_im1206'

if not os.path.exists(save_root):
    os.makedirs(save_root)

while True:
    fn = '%simg%5d.bmp' % ('s8', s_num)
    anno_path = os.path.join(im_root, fn)
    #pdb.set_trace()
    
    if os.path.exists(anno_path):
        f = True
        prev_path = []
        prev_im = []
        for dt in range(-inter_num + 1, 0):
            t = s_num + dt*40
            fn_ = '%simg%05d.bmp' % ('s8', t)
            im_path = os.path.join(im_root, fn_)
            fn_save = '%s_prev_%05d.bmp' % (fn, t)
            save_path = os.path.join(save_root, fn_save)
            #pdb.set_trace()
            if not os.path.exists(im_path):
                #pdb.set_trace()
                f = False
                break
            
            prev_path.append(save_path)
            prev_im.append(cv2.imread(im_path))
        #pdb.set_trace()

        if f:
            path = os.path.join(save_root, fn)
            im = cv2.imread(anno_path)
            cv2.imwrite(path, im)
            for i in range(len(prev_path)):
                p = prev_path[i]
                im = prev_im[i]
                cv2.imwrite(p, im)
            #pdb.set_trace()
    else:
        break

    s_num += inter_num
 
