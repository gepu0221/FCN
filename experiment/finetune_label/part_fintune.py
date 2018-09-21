import numpy as np
import cv2
import pdb
import os
from ellipse_my import ellipse_my
from watershed import watershed

w_thr = 8
h_thr = 8

def get_part_set(ellipse_info, part_num=128):
    '''
    Use ellipse_info to get part point set.
    '''
    part_label = ellipse_my(ellipse_info, part_num)

    return part_label

def get_single_box(cur_axis, next_axis):
    
    lx = np.min((cur_axis[1], next_axis[1]))
    rx = np.max((cur_axis[1], next_axis[1]))
    ly = np.min((cur_axis[0], next_axis[0]))
    ry = np.max((cur_axis[0], next_axis[0]))

    cx = (lx + rx) / 2
    cy = (ly + ry) / 2
    w = (rx - lx) / 2
    h = (ry - ly) / 2
    #print('before (%d, %d)' % (w, h))
    if w < w_thr:
        w = w_thr
    if h < h_thr:
        h = h_thr

    #print('after (%d, %d)' % (w, h))

    print('cx: %d, cy: %d' % (cx, cy))

    box = [int(cx-w), int(cy-h), int(cx+w), int(cy+h)]
    #box = [int(lx), int(ly), int(rx), int(ry)]

    return box

def get_part_box(ellipse_info, part_num=128):
    
    box_list = []

    part_set = get_part_set(ellipse_info, part_num)

    for i in range(len(part_set)):

        cur_index = i
        next_index = (i+1) % len(part_set)
        box = get_single_box(part_set[cur_index], part_set[next_index])
        box_list.append(box)

    return box_list


def threshold_(im):
    _, thresh = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY_INV)

    return thresh

def adaptiveThreshold_(im):
    
    #cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

    return im

def watershed_im(im, im_gray, ellipse_info):
    
    box_list = get_part_box(ellipse_info)
    for i in range(len(box_list)):
        box = box_list[i]
        im_crop = im[box[1]:box[3], box[0]:box[2]]
        watered_crop = watershed(im_crop)
        #watered_crop = threshold_(im_crop)
        im_gray[box[1]:box[3], box[0]:box[2]] = watered_crop

    return im_gray

def adpThreshold_im(im, im_gray, ellipse_info):
    
    box_list = get_part_box(ellipse_info)
    im_gray_c = im_gray.copy()

    for i in range(len(box_list)):
        box = box_list[i]
        im_crop = im_gray[box[1]:box[3], box[0]:box[2]]
        crop = adaptiveThreshold_(im_crop)
        ii = 0
        jj = 0
        
        for i in range(box[1], box[3]):
            for j in range(box[0], box[2]):
                #print('ii: %d, jj: %d' % (ii, jj))
                #print('i: %d, j:%d' % (i, j))
                if crop[ii][jj] == 0:
                    im_gray_c[i][j] = 0
                jj += 1
            ii += 1
            jj = 0
        #im_gray_c[box[1]:box[3], box[0]:box[2]] = watered_crop

    return im_gray_c


def main():
    filename = 'img00001'
    im = cv2.imread('%s.bmp' % filename)
    
    lx, ly, rx, ry = 856, 261, 1346, 730
    w=rx-lx
    h=ry-ly
    cx=(lx+rx)/2
    cy=(ly+ry)/2

    ellipse_info = ((cx, cy), (w, h), 0)
    print(ellipse_info)
    box_list = get_part_box(ellipse_info)
    cv2.ellipse(im, ellipse_info, (0,255,0),1)

    for i in range(len(box_list)):
        box = box_list[i]
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),  (0,255,0), 1)

    cv2.imwrite('%s_re_box.bmp' %  filename, im)

def main_watershed():

    filename = 'img00004'
    im = cv2.imread('%s.bmp' % filename)
    im_gray = cv2.imread('%s.bmp' % filename, 0)
    
    lx, ly, rx, ry = 856, 261, 1346, 730
    w=rx-lx
    h=ry-ly
    cx=(lx+rx)/2
    cy=(ly+ry)/2

    ellipse_info = ((cx, cy), (w, h), 0)
    after_watershed = watershed_im(im, im_gray, ellipse_info)

    cv2.imwrite('%s_watershed.bmp' % filename, after_watershed)

def main_adap():

    filename = 'img00004'
    im = cv2.imread('%s.bmp' % filename)
    im_gray = cv2.imread('%s.bmp' % filename, 0)
    
    lx, ly, rx, ry = 856, 261, 1346, 730
    w=rx-lx
    h=ry-ly
    cx=(lx+rx)/2
    cy=(ly+ry)/2

    ellipse_info = ((cx, cy), (w, h), 0)
    after_adap = adpThreshold_im(im, im_gray, ellipse_info)

    cv2.imwrite('%s_watershed.bmp' % filename, after_adap)

def main_test_adap():
    
    filename = 'img00004'
    im = cv2.imread('%s.bmp' % filename)
    im_gray = cv2.imread('%s.bmp' % filename, 0)

    im = adaptiveThreshold_(im_gray)

    cv2.imwrite('adap_%s.bmp' % filename, im)


if __name__ == '__main__':
    #main()
    #main_watershed()
    main_adap()
    #main_test_adap()

