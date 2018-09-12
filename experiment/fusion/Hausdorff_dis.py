import numpy as np
from ellipse_my import ellipse_my_pt

def cal_haus_dis(pred, gt):
    '''
    Calculate hausdorff distance between set1 and set2.
    Args:
        set1: im shape of set1 [im_sz, im_sz, 1, 2]
        set2: gt shape of set2 [1, 1, gt_num, 2]
    '''
    im_sz = im.shape[0]
    gt_num = gt.shape[2]

    normalized_x = np.tile(im, [1, 1, gt_num, 1])
    normalized_y = np.tile(gt, [im_sz, im_sz, 1, 1])

    differences = np.subtract(normalized_x, normalized_y)
    d_matrix = np.sum(np.power(differences, 2), 3)

    term_1 = np.sum(d_matrix.min(2)) / (im_sz*im_sz)
    term_2 = np.sum(d_matrix.min(0,1)) / gt_num

    dis = term_1 + term_2

    return dis

def cal_haus_dis_p(set1, set2):
    '''
    Calculate hausdorff distance between points set1 and points set2.
    Args:
        set1: im shape of set1 [set1_num, 2]
        set2: gt shape of set2 [set2_num, 2]
    '''
    set1 = np.array(set1)
    set2 = np.array(set2)

    num_1 = set1.shape[0]
    num_2 = set2.shape[0]

    set1 = np.expand_dims(set1, axis=0)
    set2 = np.expand_dims(set2, axis=1)

    normalized_x = np.tile(set1, [num_2, 1, 1])
    normalized_y = np.tile(set2, [1, num_1, 1])

    #print('n_x', normalized_x)
    #print('n_y', normalized_y)
    differences = np.subtract(normalized_x, normalized_y)
    d_matrix = np.sum(np.power(differences, 2), 2)
    #print('diff', differences)
    #print('d_m', d_matrix)

    term_1 = np.sum(d_matrix.min(1)) / num_2
    term_2 = np.sum(d_matrix.min(0)) / num_1

    #print(term_1)
    #print(term_2)
    dis = term_1 + term_2

    return dis



def cal_haus_loss(pred_ellip, gt_ellip):
    '''
    Calculate hausdorff distance of pred ellipse and gt_ellipse
    '''
    pred_pt = ellipse_my_pt(pred_ellip)
    gt_pt = ellipse_my_pt(gt_ellip)

    haus_dis = cal_haus_dis_p(pred_pt, gt_pt)


    return haus_dis
    
def main():

    a = np.random.rand(1,2) * 10
    b = np.random.rand(2,2) * 10
    print(a)
    print(b)
    dis = cal_haus_dis_p(a, b)

    print(dis)

if __name__ == '__main__':
    main()
