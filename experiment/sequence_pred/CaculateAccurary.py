import numpy as np
import cv2
import time 
import pdb
import os
from generate_heatmap import density_heatmap

try:
    from .cfgs.config_acc import cfgs
except Exception:
    from cfgs.config_acc import cfgs



def caculate_accurary(pred_annotation,annotation):
    sz_=pred_annotation.shape
    #the number of prediction label 1
    pred_p_num=0
    #the number of correct prediction label 1
    pred_p_c_num=0
    #the number of label 1
    anno_num=0
    for i in range(sz_[0]):
        for j in range(sz_[1]):
            for k in range(sz_[2]):
                if pred_annotation[i][j][k][0]==1:
                    pred_p_num=pred_p_num+1
                    if annotation[i][j][k][0]==1:
                        pred_p_c_num=pred_p_c_num+1
                if annotation[i][j][k][0]==1:
                    anno_num=anno_num+1
    #numerator/denominator   
    #iou_accurary
    iou_deno=anno_num+pred_p_num-pred_p_c_num
    if iou_deno==0:
        accu_iou=0
    else:
        accu_iou=pred_p_c_num/iou_deno*100
        
    #pixel accuracy
    if anno_num==0:
        accu_pixel=0
    else:
        accu_pixel=pred_p_c_num/anno_num*100
    #print('Caculate accurary...')
    print('pred_p_c_num:%d, iou_deno:%d, anno_num:%d' % (pred_p_c_num, iou_deno, anno_num))

    return accu_iou, accu_pixel

#prob_thresh: the threshold of probability
def caculate_soft_accurary(pred_anno, anno, prob_thresh):
    sz_ = pred_anno.shape
    pred_p_num = 0
    pred_p_c_num = 0
    anno_num = 0
    for i in range(sz_[0]):
        for j in range(sz_[1]):
            for k in range(sz_[2]):
                if pred_anno[i][j][k][1] >= prob_thresh:
                    pred_p_num += 1
                    if anno[i][j][k][1] == 1:
                        pred_p_c_num += 1
                if anno[i][j][k][1] == 1:
                    anno_num += 1

    #numerator/denominator   
    #iou_accurary
    iou_deno=anno_num+pred_p_num-pred_p_c_num
    print('pred_p_c_num:%d, iou_deno:%d, anno_num:%d' % (pred_p_c_num, iou_deno, anno_num))
    if iou_deno==0:
        accu_iou=0
    else:
        accu_iou=pred_p_c_num/iou_deno*100
        
    #pixel accuracy
    if anno_num==0:
        accu_pixel=0
    else:
        accu_pixel=pred_p_c_num/anno_num*100
    
    return accu_iou,accu_pixel 

#Create ellipse error related folder.
def create_ellipse_f():
    train_error_path = cfgs.error_path
    valid_error_path = cfgs.error_path+'_valid'
    if not os.path.exists(train_error_path):
        os.makedirs(train_error_path)
    if not os.path.exists(valid_error_path):
        os.makedirs(valid_error_path)

#generate ellipse to compare ellispes info
def caculate_ellip_accu_once(im, filename, pred, pred_pro, gt_ellip, if_valid=False):
#gt_ellipse [(x,y), w, h]
    pts = []
    #pdb.set_trace()
    _, p, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(p)
    #pdb.set_trace()
    
    
    for i in range(len(p)):
        for j in range(len(p[i])):
            pts.append(p[i][j])
    #pdb.set_trace()
    pts_ = np.array(pts)
    if pts_.shape[0] > 5:
        ellipse_info = cv2.fitEllipse(pts_)
        pred_ellip = np.array([ellipse_info[0][0], ellipse_info[0][1], ellipse_info[1][0], ellipse_info[1][1]])
        ellipse_info = (tuple(np.array([ellipse_info[0][0], ellipse_info[0][1]])), tuple(np.array([ellipse_info[1][0], ellipse_info[1][1]])), 0)
    else:
        pred_ellip = np.array([0,0,0,0])
        ellipse_info = (tuple(np.array([0,0])), tuple(np.array([0,0])), 0)
    
    loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2))
    #save worse result
    if if_valid:
        error_path = cfgs.error_path+'_valid'
    else:
        error_path = cfgs.error_path

    if loss > cfgs.loss_thresh:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.ellipse(im,ellipse_info,(0,255,0),1)
        gt_ellip_info = (tuple(np.array([gt_ellip[0], gt_ellip[1]])), tuple(np.array([gt_ellip[2], gt_ellip[3]])), 0)
        cv2.ellipse(im,gt_ellip_info,(0,0,255),1)
        path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'.bmp')
        cv2.imwrite(path_, im)

        #heatmap
        heat_map = density_heatmap(pred_pro[:, :, 1])
        cv2.imwrite(os.path.join(error_path, filename.strip().decode('utf-8')+'_heatseq_.bmp'), heat_map)



    return loss

def caculate_ellip_accu(im, filenames, pred, pred_pro, gt_ellip, if_valid=False):
    sz_ = pred.shape
    loss = 0
    for i in range(sz_[0]):
        loss += caculate_ellip_accu_once(im[i], filenames[i], pred[i].astype(np.uint8), pred_pro[i], gt_ellip[i], if_valid)

    return loss
