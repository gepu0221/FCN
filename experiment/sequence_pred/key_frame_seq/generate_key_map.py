import numpy as np
import cv2
import os
import pdb

try:
    from .cfgs.config_key_map import cfgs
except Exception:
    from cfgs.config_key_map import cfgs

#use ellipse loss to choose key frame.
def choose_key_frame(im, filename, pred, gt_ellip):
#gt_ellipse [(x,y), w, h]
    pts = []
    _, p, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(p)):
        for j in range(len(p[i])):
            pts.append(p[i][j])
    pts_ = np.array(pts)
    if pts_.shape[0] > 5:
        ellipse_info = cv2.fitEllipse(pts_)
        pred_ellip = np.array([ellipse_info[0][0], ellipse_info[0][1], ellipse_info[1][0], ellipse_info[1][1]])
        ellipse_info = (tuple(np.array([ellipse_info[0][0], ellipse_info[0][1]])), tuple(np.array([ellipse_info[1][0], ellipse_info[1][1]])), 0)
    else:
        pred_ellip = np.array([0,0,0,0])
        ellipse_info = (tuple(np.array([0,0])), tuple(np.array([0,0])), 0)
    
    loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2))

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.ellipse(im,ellipse_info,(0,255,0),1)
    gt_ellip_info = (tuple(np.array([gt_ellip[0], gt_ellip[1]])), tuple(np.array([gt_ellip[2], gt_ellip[3]])), 0)
    cv2.ellipse(im,gt_ellip_info,(0,0,255),1)

    #save worse result
    if_key = True
    if loss > cfgs.loss_thresh:
        if_key = False
        path_ = os.path.join(cfgs.error_path, filename.strip().decode('utf-8')+'.bmp')

    else:
        path_ = os.path.join(cfgs.correct_path, filename.strip().decode('utf-8')+'.bmp')
    cv2.imwrite(path_, im)



    return if_key


