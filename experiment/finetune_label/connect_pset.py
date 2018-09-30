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
    c_list = connect(pset)
    pdb.set_trace()

if __name__ == '__main__':
    main()
        


