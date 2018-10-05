import numpy as np
import pdb
import cv2

def get_dis(p1, p2):
    
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    dis = np.sqrt(dis)
 
    return dis

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
        min_ix = pset
        for i in range(len(pset)):
            p_ = pset[i]
            dis = get_dis(p, p_)
            if dis < min_dis:
                min_dis = dis
                min_ix = p_
        #pdb.set_trace()
        c_list.append(min_ix)
        pset.remove(min_ix)
        p = min_ix


    return c_list


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
        v_ = cal_speed(c_list[len(c_list)-1], min_ix)
        #pdb.set_trace()
        c_list.append(min_ix)
        v_list.append(v_)
        pset.remove(min_ix)
        p = min_ix


    return c_list, v_list


def cal_speed(p1, p2):
    
    v_x = p2[0] - p1[0]
    v_y = p2[1] - p1[1]

    #return [v_x, v_y]
    v = v_x**2 + v_y**2

    return v 

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
        


