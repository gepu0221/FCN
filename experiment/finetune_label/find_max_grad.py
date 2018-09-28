import numpy as np
import cv2
import time
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
    dis = np.sqrt(dis)
    #dis = np.power(dis, 0.8)
    if dis < 8:
        dis = 1
    else:
        dis = np.log(dis)
    
    #print('dis: ', dis)

    return dis

def find_max_grad_(grad_map, index):
    
    max_grad = 0
    max_index = index

    sz = grad_map.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            grad_dis = grad_map[i][j]/get_dis(index, [i,j])
            #print('%d, %d : %g' % (i, j, grad_dis))
            if max_grad < grad_dis:
                max_grad = grad_dis
                max_index = [i,j]
    #print('origin_index: ', index)
    #print('max_index: ', max_index)
    #pdb.set_trace()
    return max_index

def find_grad_(grad_crop, label_crop):
    
    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    re_crop = np.zeros((sz[0], sz[1]))
    #pdb.set_trace()
    for i in range(len(label_pset)):
        l_p = label_pset[i]

        max_grad_index = find_max_grad_(grad_crop, l_p)
        lx = max_grad_index[0]
        rx = max_grad_index[0]+1
        ly = max_grad_index[1]
        ry = max_grad_index[1]+1
        re_crop[lx:rx, ly:ry] = 255
        
    return re_crop

# Find max grad point with the same number of label_crop
def find_grad_num(grad_crop, label_crop):
    
    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    re_crop = np.zeros((sz[0], sz[1]))
    grad_crop_c = grad_crop.copy()

    for i in range(int(len(label_pset)/2)):
        index_ = np.argmax(grad_crop_c)
        max_grad_index = [int(index_/sz[0]), index_%sz[1]]
        grad_crop_c[max_grad_index[0]][max_grad_index[1]] = 0
     
        lx = max_grad_index[0]
        rx = max_grad_index[0]+1
        ly = max_grad_index[1]
        ry = max_grad_index[1]+1
        re_crop[lx:rx, ly:ry] = 255
        
    return re_crop


def closed_(im, num):
    kernel = np.ones((3,3), np.uint8)
    for i in range(num):
        im = cv2.dilate(im, kernel)
    for j in range(num):
        im = cv2.erode(im, kernel)

    return im

def opened_(im, num):
    kernel = np.ones((3,3), np.uint8)
    for i in range(num):
        im = cv2.erode(im, kernel)
    for j in range(num):
        im = cv2.dilate(im, kernel)

    return im

def polyfit_(im):
    
    sz = im.shape

    x = []
    y = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j] == 255:
                x.append(i)
                y.append(j)

    curve = np.polyfit(np.array(x), np.array(y), 5)
    pdb.set_trace()

def result_show(im_show, im_cc, crop, box, im_crop):
    
    ii = 0
    jj = 0
    flag = 255
    for i in range(box[1], box[3]):
        for j in range(box[0], box[2]):
            
            if crop[ii][jj] == flag:
                im_cc[i][j] = 255
            jj += 1
        ii += 1
        jj = 0
    for i in range(len(im_crop)):
        im_show[box[1]:box[3], box[0]:box[2]] = im_crop[i]


# Find grad/dis max point.
def find_grad_im(im, im_gray, ellipse_info):
    
    box_list = get_part_box(ellipse_info)
    im_c = im.copy()
    sz = im_gray.shape
    im_cc = np.zeros((sz[0], sz[1]))
    im_show = im_gray.copy()

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),2)

    for i in range(len(box_list)):
        box = box_list[i]
        
        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])
        crop = find_grad_(im_crop, ellipse_crop)
        
        
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

    for i in range(sz[0]):
        for j in range(sz[1]):
            
            if im_cc[i][j] == 255:
                im_c[i][j] = 255

    return im_c, im_show


# Use different scale to find the crop which gets most grad/dis points
def get_most_crop(crop_set):
    
    max_sum = -1
    max_index = 0
    for i in range(len(crop_set)):
        crop = crop_set[i]
        sum_ = np.sum(crop)
        if sum_ > max_sum:
            max_index = i
            max_sum = sum_
        cv2.imwrite('crop_patch/%d.bmp' % i, crop)
    pdb.set_trace()
    return crop_set[max_index], max_index

def find_grad_adap_im(im, im_gray, ellipse_info):
    
    box_list = get_adaptive_box(ellipse_info, part_num=128)
    sz = im_gray.shape

    im_c = im.copy() # Image used to show rgb with contours.
    im_cc = np.zeros((sz[0], sz[1])) # Image used to see only contours.
    im_show = im_gray.copy() # Image to show grad map.

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255), 2)

    time_all = time.time()
    for i in range(len(box_list)):
        box_set = box_list[i]
        crop_set = []
        im_crop_set = []
        time_one_box_list = time.time()
        for j in range(len(box_set)):
            box = box_set[j]
            
            if box[0]<0 or box[1]<0 or box[2]>=sz[1] or box[3]>=sz[0]:
                continue

            ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
            im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])
            crop = find_grad_(im_crop, ellipse_crop)

            crop_set.append(crop)
            im_crop_set.append(im_crop)

        crop, max_index = get_most_crop(crop_set)
        print('box_list%d: choosen index: %d' % (i, max_index))
        result_show(im_show, im_cc, crop, box_set[max_index], im_crop_set)
        print('time of a box list %g' % (time.time()-time_one_box_list)) 

    print('time of a image: %g' % (time.time()-time_all))
    for i in range(sz[0]):
        for j in range(sz[1]):
            
            if im_cc[i][j] == 255:
                im_c[i][j] = 255

    return im_c, im_show



