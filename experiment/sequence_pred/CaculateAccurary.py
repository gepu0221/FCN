import numpy as np

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
