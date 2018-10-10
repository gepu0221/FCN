import numpy as np
import pdb
import cv2
from utils import *

def get_dis(p1, p2):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
 
    return dis

#1. 
def connect(pset):
    
    '''
    Find nearest point to connect whole pset.
    Args:
        pset is a list.
    '''
    #connect list
    c_list = []
    p = pset.pop()
    c_list.append(p)
    while(len(pset) != 0):

        min_dis = 200000
        min_ix = p
        for i in range(len(pset)):
            p_ = pset[i]
            dis = get_dis(p, p_)
            if dis < min_dis:
                min_dis = dis
                min_ix = p_
        #pdb.set_trace()
        #print(min_dis)
        c_list.append(min_ix)
        pset.remove(min_ix)
        p = min_ix


    return c_list

#2. Connect obey direciton
def if_correct_dire(p1, p2, p_c, thresh):
    '''
    Judge the direction of next point if correct.
    Args:
        p_c: axis of center point.
    '''
    p1_ = np.array(p1)
    p2_ = np.array(p2)
    p_c_ = np.array(p_c)
    p_m = (p1_ + p2_) / 2
    v1 = p1_ - p2_
    v2 = p_m - p_c_
    dis1 = get_dis_true(p1, p2)
    dis2 = get_dis_true(list(p_m), p_c)
    #print('cos1: dis1: %g, dis2: %g' % (dis1, dis2))
    cos1 = np.abs((np.dot(v1, v2)) / (dis1*dis2))
 
    v1 = p1_ - p_c_
    v2 = p2_ - p_c_
    dis1 = get_dis_true(p1, p_c)
    dis2 = get_dis_true(p2, p_c)
    #print('cos2: dis1: %g, dis2: %g' % (dis1, dis2))
    cos2 = (np.dot(v1, v2)) / (dis1*dis2)
    #print('cos1: ', cos1)
    #print('cos2: ', cos2)

    flag = True
    #if cos1 > thresh or cos2 < 0.95:
    if cos1 > thresh or cos2 < 0.9:
        flag = False
    #print('flag: ', flag)
    return flag


def connect_obey_dire(pset, center):
    '''
    Connect points as colckwise or counterclockwise direction, 
    and center is contours center
    '''
    #connect_list
    c_list = []
    p = pset.pop()
    p_s = p
    c_list.append(p)
    while(len(pset) != 0):
        
        min_dis = 200000
        min_ix = p
        for i in range(len(pset)):
            p_ = pset[i]
            dis = get_dis2_true(p, p_)
            if dis < min_dis:
                min_dis = dis
                min_ix = p_
        if if_correct_dire(p, min_ix, center, 0.2):
            #if min_dis < 30:
            c_list.append(min_ix)
            p = min_ix
        pset.remove(min_ix)
    
    #pdb.set_trace()
    c_list.append(p_s)
    p_e = c_list[len(c_list)-2]

    return c_list, p_s, p_e

def polar_transform(p, if_r=False):
    
    angle = np.arctan2(p[1], p[0])
    r = 0
    if if_r:
        r = np.sqrt(np.power(p[1], 2) + np.power(p[0], 2))

    return angle, r


def connect_obey_polar(pset, center, dire_thresh=0.2, polar_off = 0.01, r_off = 5):
    '''
    Connect points until return to thr near polar angle the starting point.
    '''
    c_list = []
    p = pset.pop()
    p_s = p
    #calculate polar angle.
    s_angle, s_r = polar_transform(p_s, if_r=True)
    c_list.append(p)
    count = 0
    
    #print('polar_off: %g, r_off: %g' % (polar_off, r_off))
    while True:
        
        min_dis = 200000
        min_idx = p
        for i in range(len(pset)):
            p_ = pset[i]
            dis = get_dis2_true(p, p_)
            if dis < min_dis:
                min_dis = dis
                min_idx = p_
        #print('min_dis: %g' % min_dis)
        #print('min_idx: ', min_idx)
        #print('p: ', p)
        #pdb.set_trace()
        if if_correct_dire(p, min_idx, center, dire_thresh):
            c_list.append(min_idx)
            count += 1
            p = min_idx
            #Test polar angle if return to starting point.
            cur_angle, cur_r = polar_transform(min_idx, if_r=True)
            angle_off = np.abs(cur_angle - s_angle)
            cur_r_off = np.abs(cur_r - s_r)
            #print('angle_off: ', angle_off)
            #print('cur_r_off: ', cur_r_off)
            #print('count: ', count)
            if angle_off < polar_off and cur_r_off < r_off and count > 100:
                break
        pset.remove(min_idx)
        #if no points in pset
        if len(pset) == 0:
            break
        #pdb.set_trace()
        #print('--------------------')
    c_list.append(p_s)
    p_e = c_list[len(c_list)-2]

    return c_list, p_s, p_e

def connect_speed(pset):
    
    '''
    Find nearest point to connect whole pset.
    Args:
        pset is a list.
    '''
    #connect list
    c_list = []
    # speed list
    v_list = []
    v_dire_list = []
    p = pset.pop()
    c_list.append(p)

    while(len(pset) != 0):

        min_dis = 200000
        min_ix = pset
        for i in range(len(pset)):
            p_ = pset[i]
            dis = get_dis(p, p_)
            if dis < min_dis:
                min_dis = dis
                min_ix = p_
        #pdb.set_trace()
        pre_ix = c_list[len(c_list)-1]
        v_, v_dire_ = cal_speed(c_list[len(c_list)-1], min_ix)
        #pdb.set_trace()
        c_list.append(min_ix)
        v_list.append(v_)
        v_dire_list.append(v_dire_)
        pset.remove(min_ix)
        p = min_ix


    return c_list, v_list, v_dire_list


def cal_speed(p1, p2):
    
    v_x = p2[0] - p1[0]
    v_y = p2[1] - p1[1]

    #return [v_x, v_y]
    v = v_x**2 + v_y**2
    if v_y == 0:
        v_dire = 0
    else:
        v_dire = np.abs(v_x / v_y)

    return v, v_dire

#connect line set with line
def connect_line(c_list, im, thresh=30):
    
    p_s = c_list[0]
    p1 = c_list[0]
    p_lined = c_list[0]
    for i in range(1, len(c_list)):
        p2 = c_list[i]
        dis = get_dis(p1, p2)
        #if dis < thresh:
        if True:
            cv2.line(im, (p1[1], p1[0]), (p2[1], p2[0]), (255,255,255), 1)
            p_lined = p2
        p1 = p2
        #cv2.imwrite('line_im.bmp', im)
        #pdb.set_trace()
    if dis >= thresh:
        dis = get_dis(p_s, p_lined)
        if dis < thresh:
            cv2.line(im, (p_s[1], p_s[0]), (p_lined[1], p_lined[0]), (255,255,255), 1)
        
    return im

def main():

    pset = []
    a = [1,2]
    pset.append(a)
    a = [4,5]
    pset.append(a)
    a = [0,0]
    pset.append(a)
    a= [2,2]
    pset.append(a)
    a = [4,7]
    pset.append(a)
    
    #pdb.set_trace()
    c_list, v_list = connect_speed(pset)
    pdb.set_trace()

if __name__ == '__main__':
    main()
        


