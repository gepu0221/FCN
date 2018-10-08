import numpy as np
import cv2
import math
import time
import pdb
import os
from ellipse_my import ellipse_my
from part_fintune import *
from connect_pset import *

max_dis = 12
max_dire = 2

#1. Get point
def get_point_set(im, flag=0):
    
    sz = im.shape
    #cv2.imwrite('patch.bmp', im)
    pset = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j] == flag:
                pset.append([i, j])
    return pset

#2. Dis function
def get_dis(p1, p2):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
    #dis = np.power(dis, 0.8)
    if dis < 8:
        dis = 1
    else:
        dis = np.log(dis)    

    return dis

def get_dis_map4(sz):

    global dis_map
    dis_map = []

    for i in range(sz[0]):
        row_map = []
        for j in range(sz[1]):
            one_map = []
            for ii in range(sz[0]):
                one_row_map = []
                for jj in range(sz[1]):
                    dis = get_dis([i, j], [ii, jj])
                    one_row_map.append(dis)
                one_map.append(one_row_map)
            row_map.append(one_map)
        dis_map.append(row_map)

    return dis_map
                    
# From distance map to find dis
def find_pair_dis(p1, p2, sz):
    #print('p1_idx: ', p1)
    #print('p2_idx: ', p2)
    dis = dis_map[p1[0]][p1[1]][p2[0]][p2[1]]

    return dis 


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

def hist_process(im, hist_sz, hist_range):
    
    hist = cv2.calcHist([im], [0], None, [hist_sz], hist_range)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    sz = hist_sz + 1
    hist_im = np.zeros([sz, sz, 3], np.uint8)
    hpt = int(0.9*sz)

    for h in range(hist_sz):
        intensity = int(hist[h]*hpt/maxVal)
        cv2.line(hist_im, (h,sz), (h, sz-intensity), (255, 0, 0), 1)

    cv2.imwrite('hist.bmp', hist_im)

    im_at = cv2.adaptiveThreshold(im, 30, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    cv2.imwrite('dis_adap.bmp', im_at)

    return hist, im_at/hist_sz

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
    #pdb.set_trace()
    #hist = cv2.calcHist([dis_map.astype(np.uint8)], [0], None, [30], [0, 30])
    hist, im_at = hist_process(dis_map.astype(np.uint8), 30, [0, 30])
    

    im1 = dis_map * im_at
    im2 = -(dis_map * (im_at-1))
    min1, max1, min_idx1, max_idx1 = cv2.minMaxLoc(im1)
    min2, max2, min_idx2, max_idx2 = cv2.minMaxLoc(im2)
    off1 = max1 - min1
    off2 = max2 - min2
    off = np.abs(off1 - off2)
    #print('min1: %g, max1: %g, off=%g' % (min1, max1, max1-min1))
    #print('min2: %g, max2: %g, off=%g' % (min2, max2, max2-min2))
    max_index1 = max_index
    flag = 255
    if off < 15:
    #if True:
       
        dis1 = find_pair_dis(max_idx1, [0,0], sz)
        dis2 = find_pair_dis(max_idx2, [0,0], sz)
        if dis1 < dis2:
            max_index1 = [max_idx1[1], max_idx1[0]]
        else:
            max_index1 = [max_idx2[1], max_idx2[0]]
    #pdb.set_trace()
    if max_index1 != max_index:
        max_index = max_index1
        flag = 1
    
    return max_index, flag




def find_grad_setnum(grad_crop, label_crop):

    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    grad_pset = []
    re_crop = np.zeros((sz[0], sz[1]))
    for i in range(len(label_pset)):
        l_p = label_pset[i]
        max_grad_index = find_max_grad_(grad_crop, l_p) 
        #pdb.set_trace()
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
        max_grad_index, flag = find_max_grad_continue(grad_crop, l_p, pre_idx, center)
        #pdb.set_trace()
        pre_idx = max_grad_index
        grad_pset.append(max_grad_index)
        grad_crop[max_grad_index[0]][max_grad_index[1]] = 0
        #re_crop[max_grad_index[0]][max_grad_index[1]] = 255
        re_crop[max_grad_index[0]][max_grad_index[1]] = flag
    

    return re_crop, label_pset, grad_pset, pre_idx

def get_local_pre_idx(pre_idx, box):
    
    #pre_idx[0] -= box[0]
    #pre_idx[1] -= box[1]
    if pre_idx[0] >= box[2]:
        pre_idx[0] = box[2] - 1
    if pre_idx[1] >= box[3]:
        pre_idx[1] = box[3] - 1
    
    pre_idx[0] -= box[0]
    pre_idx[1] -= box[1]

    return pre_idx

def get_global_pre_idx(pre_idx, box):
    
    pre_idx[0] += box[0]
    pre_idx[1] += box[1]

    return pre_idx

#3. Morphological processing
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

    curve = np.polyfit(np.array(x), np.array(y), 2)

    return curve[0]

#4. Show
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
    box_list = get_part_box(ellipse_info)
    im_c = im.copy()

    im_cc = np.zeros((sz[0], sz[1]))
    im_show = im_gray.copy()

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)

    for i in range(len(box_list)):
        
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)
        box = box_list[i]

        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])

        crop, ellipse_pset, im_pset = find_grad_setnum(im_crop, ellipse_crop)

        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)
        #cv2.imwrite('grad_dire.bmp', im_c )

        
        ii = 0
        jj = 0
        
        b_cx = int((box[2]-box[0])/2)
        b_cy = int((box[3]-box[1])/2)
        flag = 255
        for i_ in range(box[1], box[3]):
            for j_ in range(box[0], box[2]):
                
                if crop[ii][jj] == flag:
                    im_cc[i_][j_] = 255
                jj += 1
            ii += 1
            jj = 0
        
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
    box_list = get_part_box(ellipse_info)
    box_list = get_new_box_list(box_list, p_s)

    im_c = im.copy()
    im_show = im_gray.copy()
    im_cc = np.zeros((sz[0], sz[1]))

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)
    pre_idx = None
    total_pset = []

    for i in range(len(box_list)):
        box = box_list[i]
        
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)

        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])
        
        if pre_idx != None:
            pre_idx = get_local_pre_idx(pre_idx, box)
        crop, ellipse_pset, im_pset, pre_idx = find_grad_continue_setnum(im_crop, ellipse_crop, pre_idx, ellipse_info[0])
        pre_idx = get_global_pre_idx(pre_idx, box)
        #pdb.set_trace()
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)
        #cv2.imwrite('grad_dire.bmp', im_c )

        
        ii = 0
        jj = 0
        flag = 255
        for i_ in range(box[1], box[3]):
            for j_ in range(box[0], box[2]):
                
                #if crop[ii][jj] == flag:
                #    im_cc[i_][j_] = 255

                if crop[ii][jj] > 0:
                    im_cc[i_][j_] = crop[ii][jj]
                
                jj += 1
            ii += 1
            jj = 0

    for i in range(sz[0]):
        for j in range(sz[1]):
            
            #if im_cc[i][j] == 255:
                #im_c[i][j] = 255
            if im_cc[i][j] > 0:
                im_c[i][j] = im_cc[i][j]
                total_pset.append([i, j])
    
    #c_list = connect(total_pset)
    #im_c = connect_line(c_list, im_c)
    
    #im_c = closed_(im_c, 3)
    
    return im_c, im_show


def find_grad_im(im, im_gray, ellipse_info):
    
    max_sp = find_grad_im1(im, im_gray, ellipse_info)
    im_c, im_show = find_grad_im2(im, im_gray, ellipse_info, max_sp)

    return im_c, im_show
