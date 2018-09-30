import numpy as np
import cv2
import pdb
import os
from ellipse_my import ellipse_my
from watershed import watershed

w_thr = 10
h_thr = 10
offset = 10
box_off =[[0, 0], [0, offset], [0, -offset], [offset, 0], [-offset, 0]]

def get_part_set(ellipse_info, part_num=128):
    '''
    Use ellipse_info to get part point set.
    '''
    part_label = ellipse_my(ellipse_info, part_num)

    return part_label

def get_single_old_box(cur_axis, next_axis):
    
    lx = np.min((cur_axis[1], next_axis[1]))
    rx = np.max((cur_axis[1], next_axis[1]))
    ly = np.min((cur_axis[0], next_axis[0]))
    ry = np.max((cur_axis[0], next_axis[0]))

    cx = (lx + rx) / 2
    cy = (ly + ry) / 2
    w = (rx - lx) / 2
    h = (ry - ly) / 2
    if w < w_thr:
        w = w_thr
    if h < h_thr:
        h = h_thr

    print('after (%d, %d)' % (w, h))

    print('cx: %d, cy: %d' % (cx, cy))

    box = [int(cx-w), int(cy-h), int(cx+w), int(cy+h)]
    #box = [int(lx), int(ly), int(rx), int(ry)]

    return box

def get_single_box(cur_axis, next_axis):

    cx = cur_axis[1]
    cy = cur_axis[0]
    w = w_thr
    h = h_thr

    #print('after (%d, %d)' % (w, h))

    #print('cx: %d, cy: %d' % (cx, cy))

    box = [int(cx-w), int(cy-h), int(cx+w), int(cy+h)]
    #box = [int(lx), int(ly), int(rx), int(ry)]

    return box

def get_diff_single_box(cur_axis, box_num=5):

    cx = cur_axis[1]
    cy = cur_axis[0]
    w = w_thr
    h = h_thr
    box_set = []

    for i in range(box_num):
        cx_n = cx + box_off[i][0]
        cy_n = cy + box_off[i][1]
        box = [int(cx_n-w), int(cy_n-h), int(cx_n+w), int(cy_n+h)]
        box_set.append(box)


    return box_set



def get_part_box(ellipse_info, part_num=64):
    
    box_list = []

    part_set = get_part_set(ellipse_info, part_num)

    for i in range(len(part_set)):

        cur_index = i
        next_index = (i+1) % len(part_set)
        box = get_single_box(part_set[cur_index], part_set[next_index])
        box_list.append(box)

    return box_list

def get_adaptive_box(ellipse_info, part_num=128):
    
    box_list = []

    part_set = get_part_set(ellipse_info, part_num)

    for i in range(len(part_set)):
        
        cur_index = i
        box_set = get_diff_single_box(part_set[cur_index])
        box_list.append(box_set)

    return box_list

def threshold_(im):
    _, thresh = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY_INV)

    return thresh


def otsu_threshold_(im):
    
    _,thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

    return thresh


def canny_(im):
    cv2.GaussianBlur(im, ksize=(5,5), sigmaX=2)
    im = cv2.Canny(im, threshold1=127, threshold2=255)

    return im

def adaptiveThreshold_(im):
    
    #cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

    return im

def grad_(im):
    
    solelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    solely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(solelx, solely)

    return grad

def grad_direction(im):
    
    im = cv2.phase(cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(im,cv2.CV_64F, 0, 1, ksize=3), angleInDegrees=True)

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
    #im_c = im_gray.copy()
    im_c = im.copy()
    sz = im_gray.shape

    for i in range(len(box_list)):
        box = box_list[i]
        
        if box[0]<0 or box[1]<0 or box[2]>= sz[1] or box[3]>=sz[0]:
            continue
        im_crop = im_gray[box[1]:box[3], box[0]:box[2]]
        crop = adaptiveThreshold_(im_crop)
        #crop = otsu_threshold_(im_crop)
        ii = 0
        jj = 0
        
        b_cx = int((box[2]-box[0])/2)
        b_cy = int((box[3]-box[1])/2)
        flag = crop[b_cy][b_cx]
        for i in range(box[1], box[3]):
            for j in range(box[0], box[2]):
                #print('ii: %d, jj: %d' % (ii, jj))
                #print('i: %d, j:%d' % (i, j))
                if crop[ii][jj] == flag:
                    im_c[i][j] = 255
                jj += 1
            ii += 1
            jj = 0
        '''
        crop3 = np.zeros((crop.shape[0], crop.shape[1], 3))
        crop3[:,:,0] = crop
        crop3[:,:,1] = crop
        crop3[:,:,2] = crop
        im_c[box[1]:box[3], box[0]:box[2]] = crop3
        '''
    return im_c


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

    #im = adaptiveThreshold_(im_gray)
    im = otsu_threshold_(im_gray)

    cv2.imwrite('adap_%s.bmp' % filename, im)

def main_test_grad():
    
    filename = 'img00003'
    im = cv2.imread('%s.bmp' % filename)
    im_gray = cv2.imread('%s.bmp' % filename, 0)

    im = grad_(im_gray)

    cv2.imwrite('grad_%s.bmp' % filename, im)

def main_test_grad_dire():
    
    filename = 'img00003'
    im = cv2.imread('%s.bmp' % filename)
    im_gray = cv2.imread('%s.bmp' % filename, 0)

    im = grad_direction(im_gray)

    cv2.imwrite('grad_dire%s.bmp' % filename, im)

if __name__ == '__main__':
    #main()
    #main_watershed()
    #main_adap()
    main_test_grad()
    #main_test_grad_dire()

