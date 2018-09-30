import numpy as np
import cv2
import time
import pdb
import os
from ellipse_my import ellipse_my
from part_fintune import *

def get_point_set(im, flag=0):
    
    sz = im.shape
    #cv2.imwrite('patch.bmp', im)
    pset = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j] == flag:
                pset.append([i, j])
    return pset

def get_dis(p1, p2):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
    #dis = np.power(dis, 0.8)
    if dis < 8:
        dis = 1
    else:
        dis = np.log(dis)    

    return dis

def get_dis_map(sz):
    
    global dis_map
    dis_map = []

    num = sz[0] * sz[1]
    for i in range(num):
        one_map = []
        main_p = [i / sz[1], i % sz[1]]
        for j in range(num):
            p = [j / sz[1], j % sz[1]]
            dis = get_dis(main_p, p)
            one_map.append(dis)
        dis_map.append(one_map)
    
    return dis_map

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

def find_grad_(grad_crop, label_crop):
    
    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    re_crop = np.zeros((sz[0], sz[1]))
    for i in range(len(label_pset)):
        l_p = label_pset[i]
        max_grad_index = find_max_grad_(grad_crop, l_p) 
        
        print('max_index: ', max_grad_index)
        re_crop[max_grad_index[0]][max_grad_index[1]] = 255
        

    return re_crop

def find_grad_setnum(grad_crop, label_crop):

    sz = grad_crop.shape
    label_pset = get_point_set(label_crop)
    re_crop = np.zeros((sz[0], sz[1]))
    for i in range(len(label_pset)):
        l_p = label_pset[i]
        max_grad_index = find_max_grad_(grad_crop, l_p) 
        #grad_crop[max_grad_index[0]][max_grad_index[1]] = 0
        re_crop[max_grad_index[0]][max_grad_index[1]] = 255

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
    cv2.ellipse(im_ellip, ellipse_info, (0,0,255),1)

    for i in range(len(box_list)):
    #for i in range(1):
        
        box = box_list[i]
        
        #cv2.rectangle(im_c, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)

        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        ellipse_crop = im_ellip[box[1]:box[3], box[0]:box[2], 0]
        im_crop = grad_(im_gray[box[1]:box[3], box[0]:box[2]])
        #crop = find_grad_(im_crop, ellipse_crop)
        crop = find_grad_setnum(im_crop, ellipse_crop)
        
        
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

    #im_cc = closed_(im_cc, 2)
    
    for i in range(sz[0]):
        for j in range(sz[1]):
            
            if im_cc[i][j] == 255:
                im_c[i][j] = 255
   
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



