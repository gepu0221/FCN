import numpy as np
import cv2
import math
import time
import pdb
import os
from ellipse_my import ellipse_my
from part_fintune import *
from connect_pset import *

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
def get_dis(p1, p2, min_dis=8):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
    #dis = np.power(dis, 0.8)
    if dis < min_dis:
        dis = 1
    else:
        dis = np.log(dis)    

    return dis

def get_dis2_true(p1, p2):

    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    if dis == 0:
        dis =0.001

    return dis

def get_dis_true(p1, p2):

    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
    
    if dis == 0:
        dis =0.001


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

def find_pair_dis_(p1, p2, sz):
    print('p1_idx: ', p1)
    print('p2_idx: ', p2)
    dis = dis_map[p1[0]][p1[1]][p2[0]][p2[1]]
    pdb.set_trace()
    return dis 



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

def dilate_(im, num):
    kernel = np.ones((3,3), np.uint8)
    for i in range(num):
        im = cv2.dilate(im, kernel)

    return im

def erode_(im, num):
    kernel = np.ones((3,3), np.uint8)
    for i in range(num):
        im = cv2.erode(im, kernel)

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

def Label_point(im, p, l, color, if_reverse=False):
    '''
    Args:
        l: label length.
        color: label color.
        if_turn: point axis if reverse
    '''
    if if_reverse:
        i = p[1]
        j = p[0]
    else:
        i = p[0]
        j = p[1]
        
    l_i = i-l
    r_i = i+l+1
    l_j = j-l
    r_j = j+l+1
    #print('l: ', l)
    #print('i: %d, %d-%d, j: %d, %d-%d' % (i, l_i, r_i, j, l_j, r_j))
    
    im[l_i:r_i, l_j:r_j, 0:3] = color

def LabelSpecificPoint(im_src, im_dst, flag, color, l, if_single_channel=-1):
    '''
    Check location in src_im which is flag.
    Label it in dst_im using re_color and length(l).
    Args:
        if_single_channel: if it >0, choose specific channel to jugde if flag
    '''
    if if_single_channel<0 or if_single_channel >2:
        raise Exception('Channel Error!!')
    ch = if_single_channel
    sz = im_src.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im_src[i][j][ch] == flag:
                Label_point(im_dst, [i,j], l, color)

def LabelPlistPoint(pset, dst_im, color, l, if_reverse=False):
    '''
    Label points in pset(list) to dst_im using color.
    '''
    
    for i in range(len(pset)):
        p = pset[i]
        #print('p: ', p)
        Label_point(dst_im, p, l, color, if_reverse)
        #pdb.set_trace()

#5. ellipse to circle transform
def polar_transform(p, center):
    #numpy axis
    cx = center[0]
    cy = center[1]
    off0 = p[0] - cx
    off1 = p[1] - cy
    off_sum = np.sqrt(off0**2 + off1**2)
    cos = off0/off_sum
    sin = off1/off_sum
    r = np.sqrt(np.power(off0, 2) + np.power(off1, 2))
    
    return cos, sin, r

def ellip_to_circle(p, w, h, center):
    '''
    Transform ellipse axis to circle axis.
    '''
    if w<h:
        normal_r = int(w/2)
    else:
        normal_r = int(h/2)

    cos, sin, r = polar_transform(p, center)
    #pdb.set_trace()
    p_cir = [0, 0]
    p[1] = int(center[0] + cos*normal_r)
    p[0] = int(center[1] + sin*normal_r)

    return p

def ellip_to_circle_nearby(p, rate1, rate2, center):
    '''
    Transform points in ellipse or nearby axis to circle axis.
    '''

    cos, sin, r = polar_transform(p, center)

    p_cir = [0, 0]
    off0 = cos*r*rate1
    off1 = sin*r*rate2
    p_cir[1] = int(center[0] + cos*r*rate1)
    p_cir[0] = int(center[1] + sin*r*rate2)
    #print('cos: %g, sin: %g, r: %g' % (cos, sin, r))
    #print('off0: %g, off1: %g' % (off0, off1))
    #print('p_cir: ', p_cir)
    #print('p', p)

    return p_cir


def ellip_cir(ellip_pset, ellipse_info):
    #opencv axis
    center = [ellipse_info[0][0], ellipse_info[0][1]]
    w = ellipse_info[1][0]
    h = ellipse_info[1][1]

    if w<h:
        normal_r2 = w
    else:
        normal_r2 = h

    rate1 = normal_r2 / w
    rate2 = normal_r2 / h
    #print('w: %d, h: %d' % (w, h))
    #print('rate1: %g, rate2: %g' % (rate1, rate2))
    #pdb.set_trace()
    cir_pset = []
    for i in range(len(ellip_pset)):
        #numpy
        p_nu = ellip_pset[i]
        #opencv
        p_o = [p_nu[1], p_nu[0]]
        p_new = ellip_to_circle_nearby(p_o, rate1, rate2, center)
        #print('p_old: ', p_nu, ' p_new: ', p_new)
        cir_pset.append([p_nu, p_new])

    return cir_pset



def main_ec():
    im = np.ones((400, 400, 3)) * 127
    #numpy cx=20, cy =10, w=20, h =10)
    center = (100, 200)
    axis = (100, 200)
    angle = 0
    ellipse_info = (center, axis, angle)
    cv2.ellipse(im, ellipse_info, (0,255,0), 1)
    cv2.imwrite('test_ec/ellipse.bmp', im)

    w = axis[0]
    h = axis[1]
    center_ = list(center)
    sz = im.shape
    p_cir_list = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j][1] == 255:
                p_ = ellip_to_circle([j, i], axis[0], axis[1], center)
                p_cir_list.append(p_)
    #pdb.set_trace()
    for i in range(len(p_cir_list)):
        p = p_cir_list[i]
        ii = p[1]
        jj = p[0]
        im[ii][jj][0:3] = np.array((255, 0, 0))     

    cv2.imwrite('test_ec/ellipse_c.bmp', im)    


#6.Local and global axis transform.
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
    pre_idx[1] -= box[0]

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



if __name__ == '__main__':
    main_ec()


