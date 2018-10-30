import numpy as np
import cv2
import os
import pdb
import glob
from utils import *
from sklearn.svm import SVC
from sklearn.externals import joblib


try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs

red = np.array((0, 0, 255))
green = np.array((0, 255, 0))
black = np.array((0, 0, 0))
white = np.array((255, 255, 255))

 
def remove_small_area(im):
    
    #pdb.set_trace()
    im_ = im[:, :, 0].astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(im_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_n = []
    for i in range(len(contours)):
        num = cv2.contourArea(contours[i])
        if num < 100:
            contours_n.append(contours[i])
    cv2.drawContours(im, contours_n, -1, (0, 0, 0), -1)
 
    return im

    

def get_occlude_pos(im, label, svc):
    
    p_list = []
    pos_list = []
    sz = im.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            p = im[i][j]
            label_p = label[i][j]
            if label_p[0] > 0:
                pos_list.append([i, j])
                p_list.append(p)

    pset = np.array(p_list)
    #pdb.set_trace()
    pred = svc.predict(pset)
    
    pred_set = []
    for i in range(len(pred)):
        if pred[i] == 1:
            pred_set.append(pos_list[i])
    #pdb.set_trace()

    return pred_set

def remove_occlude_once(im, label, svc):
    
    pos_list = get_occlude_pos(im, label, svc)

    for i in range(len(pos_list)):
        pos = pos_list[i]
        #label[pos[0]][pos[1]] = black
        Label_point(label, pos, 5, black)
    #label = opened_(label, 1)
    #label = dilate_(label, 3)
    #label = closed_(label, 2)
    #label = dilate_(label, 3)

    return label


def main():
    
    root_file = cfgs.root_file
    root_dir = '%s/Image' % root_file
    label_dir = '%s_dis4_1011' % root_file
    #label_dir = '%s_ori_label_1013' % root_file

    glob_file = os.path.join(label_dir, '*.bmp')
    file_list = []
    file_list.extend(glob.glob(glob_file))
    #save_file = '%s_dis4_1011_dilate_noOcculde_box' % root_file
    #save_file = '%s_dis4_1011_noOcculde1018' % root_file
    #save_file = '%s_ori_label_noOcculde1019' % root_file
    save_file = '%s_dis4_1011_one_pixel_noOccu_1027' % root_file


    if not os.path.exists(save_file):
        os.makedirs(save_file)
    
    #Load svc model
    if not os.path.exists(cfgs.svc_load_path):
        raise Exception('No model %s found' % cfgs.svc_load_path)
    svc = joblib.load(cfgs.svc_load_path)

    for f in file_list:
        
        fn = os.path.splitext(f.split("/")[-1])[0]
        fn_ = fn.split('img')[1]
        im_fn = os.path.join(root_dir, 'img%s%s' % (fn_, '.bmp'))

        if not os.path.exists(im_fn):
            raise Exception('No image %s found!' % im_fn)

        im = cv2.imread(im_fn)
        #Resize
        im_sz = im.shape
        im = cv2.resize(im, (int(im_sz[1]/2), int(im_sz[0]/2)), interpolation=cv2.INTER_CUBIC)

        label = cv2.imread(f)

        new_label = remove_occlude_once(im, label, svc)
        #new_label = remove_small_area(new_label)

        #Save
        '''
        save_path = os.path.join(save_file, fn+'_label3_100_closed2.bmp')
        cv2.imwrite(save_path, new_label)
        im_save_path = os.path.join(save_file, fn+'.bmp')
        cv2.imwrite(im_save_path, im)
        '''
        
        sz = im.shape
        for i in range(sz[0]):
            for j in range(sz[1]):
                if new_label[i][j][0] > 0:
                    im[i][j] = white

        show_save_path = os.path.join(save_file, fn+'_show.bmp')
        cv2.imwrite(show_save_path, im)
        
        print(f)
        save_path = os.path.join(save_file, fn+'.bmp')
        cv2.imwrite(save_path, new_label)

        #pdb.set_trace()

if __name__ == '__main__':
    main()
