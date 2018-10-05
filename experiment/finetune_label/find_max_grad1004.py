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

def polyfit_error(im):

    sz = im.shape
    x = []
    y = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j] == 255:
                x.append(i)
                y.append(j)

    curve = np.polyfit(np.array(x), np.array(y), 10)
    
    error_sum = 0
    for i in range(len(x)):
        x_ = x[i]
        error_ = np.polyval(curve, x_)
        error_sum += error_

    return error_sum

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
def comp_grad_dire(crop1, crop2, pset1, pset2):

    grad_dire1 = grad_direction(crop1)
    grad_dire2 = grad_direction(crop2)
    
    sum1 = np.zeros((2))
    sum2 = np.zeros((2))

    for i in range(len(pset1)):
        p1 = pset1[i]
        sum1 += np.array(p1)
        p2 = pset2[i]
        sum2 += np.array(p2)

    m1 = sum1/len(pset1)
    m2 = sum2/len(pset1)
    dis_sum = 0
    for i in range(len(pset2)):
        p = pset2[i]
        dis = (np.array(p)-m1)**2

        dis_sum += dis

    print(dis)
    pdb.set_trace()

def compute_dispersion(pset):
    
    sum_ = np.zeros((2))

    for i in range(len(pset)):
        p = pset[i]
        sum_ += np.array(p)
    #mean
    m = sum_/len(pset)

    dis_sum = 0
    for i in range(len(pset)):
        p = pset[i]
        dis = (np.array(p)-m)**2

        dis_sum += dis

    dis = np.sum(dis_sum)
    
    return dis


def comp_pset(crop1, crop2):
    
    k1 = polyfit_(crop1)
    k2 = polyfit_(crop2)
    tan = (k2-k1) / (1+k1*k2)
    angle = math.atan(tan)
    cos = math.cos(angle)
    #dis = np.sum((curve1 - curve2)**2)
    print(cos)
    pdb.set_trace()

def comp_pset_error(crop):
    
    error = polyfit_error(crop)
    print(error)
    pdb.set_trace()

# Find grad/dis max point.
def find_grad_im(im, im_gray, ellipse_info):
    
    sz = im_gray.shape
    box_list = get_part_box(ellipse_info)
    im_c = im.copy()
    #tmp
    im_c_grad1 = im_gray.copy()
    im_c_grad2 = im_gray.copy()
    im_c_ellip = np.zeros((sz[0], sz[1], 3))

    im_cc = np.zeros((sz[0], sz[1]))
    im_show = im_gray.copy()

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)
    cv2.ellipse(im_c_ellip, ellipse_info, (255, 0, 0), 1)

    total_pset = []
    p_map = {}
    box_num = []
    for i in range(len(box_list)):
    #for i in range(1):
        box_num.append(0)
        box = box_list[i]
        
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)

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
                    p_map['%s_%s' % (i_, j_)] = i
                    #total_pset.append([i, j])
                jj += 1
            ii += 1
            jj = 0
        im_show[box[1]:box[3], box[0]:box[2]] = im_crop

    #im_cc = closed_(im_cc, 3)

    for i in range(sz[0]):
        for j in range(sz[1]):
            
            if im_cc[i][j] == 255:
                #im_c[i][j] = 255
                total_pset.append([i, j])

    #pdb.set_trace()
    c_list, v_list, v_dire_list = connect_speed(total_pset)
    #pdb.set_trace()
    
    im_c[c_list[0][0]][c_list[0][1]] = 255
    flag = 1
    
    for i in range(len(v_list)):
        #if v_list[i] < max_dis:
        if v_dire_list[i] > max_dire and v_list[i] > max_dis:
            x = c_list[i+1][0]
            y = c_list[i+1][1]
            #pdb.set_trace()
            ix = p_map['%s_%s' % (x, y)]

            box_num[ix] += 1

    
    for i in range(len(v_list)):
        x = c_list[i+1][0]
        y = c_list[i+1][1]
        b_ix = p_map['%s_%s' % (x, y)]
        num = box_num[b_ix] 
        #pdb.set_trace()
        #if num < 1:
        if True:

            im_c[x][y] = 255
   
    return im_c, im_show

# Find grad/dis max point.(test time)
def find_grad_im_time(im, im_gray, ellipse_info):
    

    box_list = get_part_box(ellipse_info)
    sz = im_gray.shape
    im_cc = np.zeros((sz[0], sz[1]))

    im_ellip = np.ones((sz[0], sz[1], 3)) * 255
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)

    for i in range(len(box_list)):
        box = box_list[i]
        
        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        #t1 = time.time()
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])
        #t2 = time.time()
        crop = find_grad_(im_crop, ellipse_crop)        
        #t3 = time.time()
        im_cc[box[1]:box[3], box[0]:box[2]] += crop
        #t4 = time.time()
        #tt1 = float(t2 - t1)
        #tt2 = float(t3 - t2)
        #tt3 = float(t4 - t3)
        #print('%d: time_grad: %f, time_find:%f, time_process:%f' % (i, tt1, tt2, tt3))
    return im_cc




