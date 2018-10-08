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
def get_dis(p1, p2):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
    #dis = np.power(dis, 0.8)
    if dis < 8:
        dis = 1
    else:
        dis = np.log(dis)    

    return dis

def get_dis2_true(p1, p2):

    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

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


