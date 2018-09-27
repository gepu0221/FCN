import numpy as np
import cv2
import pdb
import os
from ellipse_my import ellipse_my
from part_fintune import *

def get_point_set(im, flag=0):
    
    sz = im.shape
    cv2.imwrite('patch.bmp', im)
    pset = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j] == flag:
                pset.append([i, j])
    #pdb.set_trace()
    return pset

def get_dis(p1, p2):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    return dis

def get_min_distance(pset, index):
    
    min_dis = 100000
    nearest_index = index

    for i in range(len(pset)):
        p = pset[i]
        dis = get_dis(p, index)
        if dis < min_dis:
            min_dis = dis
            nearest_index = p
    #pdb.set_trace()
    return nearest_index

def find_nearest_(adap_crop, label_crop):
    
    sz = adap_crop.shape
    label_pset = get_point_set(label_crop)
    adap_pset = get_point_set(adap_crop)
    re_crop = np.zeros((sz[0], sz[1]))

    for i in range(len(label_pset)):
        l_p = label_pset[i]
        n_index = get_min_distance(adap_pset, l_p)
        #print(n_index)
        lx = n_index[0]
        rx = n_index[0]+1
        ly = n_index[1]
        ry = n_index[1]+1
        re_crop[lx:rx, ly:ry] = 255
        #re_crop[n_index[0]][n_index[1]] = 255
    #pdb.set_trace()
    return re_crop
                

def find_nearest_im(im, im_gray, ellipse_info):

    box_list = get_part_box(ellipse_info)
    im_c = im.copy()
    sz = im_gray.shape
    im_cc = np.zeros((sz[0], sz[1]))
    im_show = im_gray.copy()

    im_adaped = adaptiveThreshold_(im_gray)
    #pdb.set_trace()
    cv2.imwrite('re_adap.bmp', im_adaped)
    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)

    for i in range(len(box_list)):
        box = box_list[i]
        
        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        #im_crop = im_adaped[box[1]:box[3], box[0]:box[2]]
        im_crop = adaptiveThreshold_(im_gray[box[1]:box[3], box[0]:box[2]])
        kernel = np.ones((5,5), np.uint8)
        
        #cv2.imwrite('before.bmp', im_crop)
        im_crop = cv2.erode(im_crop, kernel)
        im_crop = cv2.erode(im_crop, kernel)
        im_crop = cv2.dilate(im_crop, kernel)
        im_crop = cv2.dilate(im_crop, kernel)
        #cv2.imwrite('after.bmp', im_crop)
        #pdb.set_trace()
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        crop = find_nearest_(im_crop, ellipse_crop)
        #crop = find_nearest_(im_crop, )
        ii = 0
        jj = 0
        
        b_cx = int((box[2]-box[0])/2)
        b_cy = int((box[3]-box[1])/2)
        flag = 255
        for i in range(box[1], box[3]):
            for j in range(box[0], box[2]):
                
                if crop[ii][jj] == flag:
                    im_cc[i][j] = 255
                jj += 1
            ii += 1
            jj = 0
        im_show[box[1]:box[3], box[0]:box[2]] = im_crop
        '''
        crop3 = np.zeros((crop.shape[0], crop.shape[1], 3))
        crop3[:,:,0] = crop
        crop3[:,:,1] = crop
        crop3[:,:,2] = crop
        im_c[box[1]:box[3], box[0]:box[2]] = crop3
        '''
    kernel = np.ones((5,5), np.uint8)
    im_cc = cv2.dilate(im_cc, kernel)
    #im_cc = cv2.dilate(im_cc, kernel)
    im_cc = cv2.erode(im_cc, kernel)
    #im_cc = cv2.erode(im_cc, kernel)
    for i in range(sz[0]):
        for j in range(sz[1]):
            
            if im_cc[i][j] == 255:
                im_c[i][j] = 255

    return im_c, im_show


