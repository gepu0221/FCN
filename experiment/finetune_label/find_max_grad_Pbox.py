#Use cv2.ellipse to get ellipse point set, use every point as box center to process.
import numpy as np
import cv2
import math
import time
import pdb
import os
from ellipse_my import ellipse_my
from part_fintune import *
from connect_pset import *
from utils import *

max_dis = 12
max_dire = 2

                   
def find_max_grad_(grad_map, index):
    
    max_grad = 0
    max_index = index
    sz = grad_map.shape
  
    for i in range(sz[0]):
        for j in range(sz[1]):
            #dis = get_dis(index, [i,j])
            dis = find_pair_dis(index, [i,j], sz)
            #pdb.set_trace()
            grad_dis = grad_map[i][j]/dis
            if max_grad < grad_dis:
                max_grad = grad_dis
                max_index = [i,j]
 
    return max_index

# use adaptiveThreshold to process dis_map
def adap_hist_process(im, hist_sz, hist_range):
    
    hist = cv2.calcHist([im], [0], None, [hist_sz], hist_range)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    sz = hist_sz + 1
    hist_im = np.zeros([sz, sz, 3], np.uint8)
    hpt = int(0.9*sz)

    for h in range(hist_sz):
        intensity = int(hist[h]*hpt/maxVal)
        cv2.line(hist_im, (h,sz), (h, sz-intensity), (255, 0, 0), 1)

    #cv2.imwrite('hist.bmp', hist_im)

    im_at = cv2.adaptiveThreshold(im, 30, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    #cv2.imwrite('dis_adap.bmp', im_at)

    return hist, im_at/hist_sz
# use simple threshold to process dis_map
def hist_process(im, hist_sz, hist_range, thresh):
    
    hist = cv2.calcHist([im], [0], None, [hist_sz], hist_range)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    sz = hist_sz + 1
    hist_im = np.zeros([sz, sz, 3], np.uint8)
    hpt = int(0.9*sz)

    for h in range(hist_sz):
        intensity = int(hist[h]*hpt/maxVal)
        cv2.line(hist_im, (h,sz), (h, sz-intensity), (255, 0, 0), 1)

    #cv2.imwrite('hist.bmp', hist_im)

    _, im_at = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY_INV)
    #cv2.imwrite('dis_adap_inv.bmp', im_at)
    #_, im_at = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('dis_adap.bmp', im_at)
    #pdb.set_trace()
    
    return hist, im_at/255


def find_max_grad_continue(grad_map, index, pre_idx, center):
    
    max_grad = 0
    max_index = index
    sz = grad_map.shape
    
    dis_map = np.zeros((sz[0], sz[1]))

    for i in range(sz[0]):
        for j in range(sz[1]):
            dis1 = find_pair_dis(index, [i,j], sz)
            if pre_idx == None:
                dis2 = 0
            else:
                dis2 = find_pair_dis(pre_idx, [i,j], sz)
            dis = dis1 + dis2
            grad_dis = grad_map[i][j]/dis
            dis_map[i][j] = int(grad_dis)
            if max_grad < grad_dis:
                max_grad = grad_dis
                max_index = [i,j]
    hist, im_at = hist_process(dis_map.astype(np.uint8), 30, [0, 30], int(max_grad)-5)
    flag = 255
    sum_ = np.sum(im_at)
    if sum_ > 5:
        im1 = dis_map * im_at
        min1, max1, min_idx1, max_idx1 = cv2.minMaxLoc(im1)
        dis1 = get_dis2_true(max_idx1, center)
        dis2 = get_dis2_true([max_index[1], max_index[0]], center)

        if dis1 < dis2:
            max_index = [max_idx1[1], max_idx1[0]]
            flag = 1
        
    return max_index, flag, im_at*255




def find_grad_setnum(grad_crop, label_crop):

    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    grad_pset = []
    re_crop = np.zeros((sz[0], sz[1]))
    for i in range(len(label_pset)):
        l_p = label_pset[i]
        max_grad_index = find_max_grad_(grad_crop, l_p) 
        grad_pset.append(max_grad_index)
        grad_crop[max_grad_index[0]][max_grad_index[1]] = 0
        re_crop[max_grad_index[0]][max_grad_index[1]] = 255


    return re_crop, label_pset, grad_pset

def find_grad_continue_setnum(grad_crop, label_crop, pre_idx, center):

    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    grad_pset = []
    re_crop = np.zeros((sz[0], sz[1]))
    for i in range(len(label_pset)):
        l_p = label_pset[i]
        #pdb.set_trace()
        max_grad_index, flag, im_at = find_max_grad_continue(grad_crop, l_p, pre_idx, center)
        #pdb.set_trace()
        pre_idx = max_grad_index
        grad_pset.append(max_grad_index)
        grad_crop[max_grad_index[0]][max_grad_index[1]] = 0
        #re_crop[max_grad_index[0]][max_grad_index[1]] = 255
        re_crop[max_grad_index[0]][max_grad_index[1]] = flag
    

    return re_crop, label_pset, grad_pset, pre_idx, im_at

def get_local_pre_idx(pre_idx_o, box):
    
    pre_idx = pre_idx_o.copy()
    if pre_idx[0] >= box[3]:
        pre_idx[0] = box[3] - 1
    if pre_idx[1] >= box[2]:
        pre_idx[1] = box[2] - 1
    if pre_idx[0] < box[1]:
        pre_idx[0] = box[1]
    if pre_idx[1] < box[0]:
        pre_idx[1] = box[0]
    
    pre_idx[0] -= box[1]
    pre_idx[1] -= box[2]

    return pre_idx


def get_absolute_local_idx(idx_o, box):
    
    #axis type: cv2 to numpy    
    idx = idx_o.copy()
    idx[0] -= box[0]
    idx[1] -= box[1]

    return [idx[1], idx[0]]

def get_global_pre_idx(pre_idx_o, box):

    pre_idx = pre_idx_o.copy()
    
    pre_idx[0] += box[0]
    pre_idx[1] += box[1]

    return pre_idx

def get_global_idx(idx, box):
    
    idx_n = idx.copy()
    
    idx_n[0] = idx[0] + box[1]
    idx_n[1] = idx[1] + box[0]

    return idx_n



#5. grad_direction
# Find grad/dis max point.
def find_max_area(im):
    
    im_ = im.astype(np.uint8)
    cv2.imwrite('all_area.bmp', im)
    im_show = im.copy()
    sz = im.shape
    c_im_show = np.zeros((sz[0], sz[1], 3))
    _, contours, hierarchy = cv2.findContours(im_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(c_im_show, contours, -1, (0, 255, 0), 1)
    cv2.imwrite('contours.bmp', c_im_show)
    area = []
    for i in range(len(contours)):
        #area.append(cv2.contourArea(contours[i]))
        area.append(len(contours[i]))

    max_idx = np.argmax(area)
    max_area = contours[max_idx]
    #pdb.set_trace()
    max_area_s = [max_area[1][0][1], max_area[1][0][0]]
    
    for i in range(len(max_area)):
        p = max_area[i][0]
        c_im_show[p[1]][p[0]][0] = 0
        c_im_show[p[1]][p[0]][1] = 0
        c_im_show[p[1]][p[0]][2] = 255


    cv2.imwrite('area_max.bmp', c_im_show)
        
    return max_area_s


#1. Step1: after grad/dis operater, find the max connection area as the next operator start.
def find_grad_im1(im, im_gray, ellipse_info):
    
    sz = im_gray.shape
    im_c = im.copy()

    im_cc = np.zeros((sz[0], sz[1]))
    im_show = im_gray.copy()

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)
    box_list, pset = get_point_box(im_ellip, im_c)

    for i in range(len(box_list)):
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)
        box = box_list[i]
        c_idx = pset[i]

        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            #print('continue')
            continue

        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)
        #cv2.imwrite('0grad_dire.bmp', im_c )

        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])

        max_idx = find_max_grad_(im_crop, c_idx)

        g_idx = get_global_idx(max_idx, box)
        im_cc[g_idx[0]][g_idx[1]] = 255

    max_sp = find_max_area(im_cc)
    
    return max_sp



# Find grad/dis max point adding continuity.
def get_new_box_list(box_list, p_s):

    idx = 0
    new_box_list = []
    for i in range(len(box_list)):
        box = box_list[i]
        if p_s[0] >= box[0] and p_s[0] <= box[1] and p_s[1] >= box[2] and p_s[1] <= box[3]:
            idx = i
            break
    for i in range(len(box_list)):
        new_box_list.append(box_list[idx])
        idx = (idx + 1) % len(box_list)

    return new_box_list


def find_grad_im2(im, im_gray, ellipse_info, p_s):
    sz = im_gray.shape

    im_c = im.copy()
    im_show = im_gray.copy()
    im_cc = np.zeros((sz[0], sz[1]))

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)
    box_list, pset = get_point_box(im_ellip, im_c)
    box_list = get_new_box_list(box_list, p_s)
    pre_idx = None
    total_pset = []

    for i in range(len(box_list)):
     
        #box axis type: cv2
        box = box_list[i]
        c_idx = pset[i]

        center = get_absolute_local_idx([ellipse_info[0][0], ellipse_info[0][1]], box)
        
        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])
        
        if pre_idx != None:
            #print('before pre_idx: ', pre_idx)
            pre_idx = get_local_pre_idx(pre_idx, box)
        #print('c_dix: ', c_idx, ' pre_dix: ', pre_idx)
        max_idx, flag, im_at = find_max_grad_continue(im_crop, c_idx, pre_idx, center)

        g_idx = get_global_idx(max_idx, box)
        pre_idx = g_idx
        
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)
        #cv2.imwrite('grad_dire.bmp', im_c )

        im_show[box[1]:box[3], box[0]:box[2]] = im_at
        #flag = 255
        im_cc[g_idx[0]][g_idx[1]] = flag       

    for i in range(sz[0]):
        for j in range(sz[1]):
            
            #if im_cc[i][j] == 255:
                #im_c[i][j] = 255
            if im_cc[i][j] > 0:
                im_c[i][j] = im_cc[i][j]
                total_pset.append([i, j])
    
    '''
    c_list = connect(total_pset)
    im_conn = np.zeros((sz[0], sz[1], 3))
    im_conn = connect_line(c_list, im_conn)
    #im_conn = dilate_(im_conn, 2)
    im_contours = im_conn[:,:,0].astype(np.uint8)
    #Draw contours
    c_im_show = np.zeros((sz[0], sz[1], 3))
    _, contours, hierarchy = cv2.findContours(im_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(c_im_show, contours, -1, (0, 255, 0), 1)
    #cv2.imwrite('re_contours.bmp', c_im_show)
    color = np.array((255, 255, 255))
    for i in range(sz[0]):
        for j in range(sz[1]):
            if c_im_show[i][j][1] == 255:
                im_c[i-1:i+2, j-1:j+2, 0:3] = color
    '''
    return im_c, im_show


def find_grad_im(im, im_gray, ellipse_info):
    
    max_sp = find_grad_im1(im, im_gray, ellipse_info)
    im_c, im_show = find_grad_im2(im, im_gray, ellipse_info, max_sp)

    return im_c, im_show
