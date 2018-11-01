import numpy as np
import cv2
import time 
import pdb
import os
from generate_heatmap import density_heatmap
from six.moves import cPickle as pickle
from Hausdorff_dis import cal_haus_loss


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
    print('Caculate accurary...')
    #print('pred_p_c_num:%d, iou_deno:%d, anno_num:%d' % (pred_p_c_num, iou_deno, anno_num))
    print('pred_p_c_num:%d, pred_num:%d, anno_num:%d' % (pred_p_c_num, pred_p_num, anno_num))


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
    valid_error_path = cfgs.error_path+'_better'
    if not os.path.exists(train_error_path):
        os.makedirs(train_error_path)
    if not os.path.exists(valid_error_path):
        os.makedirs(valid_error_path)

#generate ellipse to compare ellispes info
def caculate_ellip_accu_once(im, filename, pred, pred_pro, gt_ellip, if_valid=False):
#gt_ellipse [(x,y), w, h]
    pts = []
    _, p, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #tmp
    sz = im.shape
    c_im_show = np.zeros((sz[0], sz[1], 3))
    cv2.drawContours(c_im_show, p, -1, (0, 255, 0), 1)
    #contours_path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'_contour.bmp')
    #cv2.imwrite(contours_path_, c_im_show)

    
    
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
    
    sz_ = im.shape
    loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2)) / (sz_[0]*sz_[1])
    
    #save worse result
    if if_valid:
        error_path = cfgs.error_path+'_valid'
    else:
        error_path = cfgs.error_path

    #tmp
    contours_path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'_contour.bmp')
    cv2.imwrite(contours_path_, c_im_show)

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

        contours_path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'_contour.bmp')
        cv2.imwrite(contours_path_, c_im_show)


    return loss

def caculate_ellip_accu(im, filenames, pred, pred_pro, gt_ellip, if_valid=False):
    sz_ = pred.shape
    loss = 0
    for i in range(sz_[0]):
        #loss += caculate_ellip_accu_once(im[i], filenames[i], pred[i].astype(np.uint8), pred_pro[i], gt_ellip[i], if_valid)
        loss += divide_shelter_once(im[i], filenames[i], pred[i].astype(np.uint8), pred_pro[i], gt_ellip[i], if_valid)


    return loss

def divide_shelter_once(im, filename, pred, pred_pro, gt_ellip, if_valid=False):
    '''
    divide the shelter and not shelter
    '''
    pts = []
    _, p, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #tmp
    sz = im.shape
    c_im_show = np.zeros((sz[0], sz[1], 3))
    cv2.drawContours(c_im_show, p, -1, (0, 255, 0), 1)

    
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
    
    loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2)) / (gt_ellip[2]*gt_ellip[3]) * 1000
  
    #save worse result
    error_path = cfgs.error_path
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.ellipse(im,ellipse_info,(0,255,0),1)
    gt_ellip_info = (tuple(np.array([gt_ellip[0], gt_ellip[1]])), tuple(np.array([gt_ellip[2], gt_ellip[3]])), 0)
    cv2.ellipse(im,gt_ellip_info,(0,0,255),1)


    if loss > cfgs.loss_thresh:
        path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'.bmp')
        cv2.imwrite(path_, im)
        #tmp
        contours_path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'_contour.bmp')
        cv2.imwrite(contours_path_, c_im_show)

    else:
        path_ = os.path.join(error_path+'_better', filename.strip().decode('utf-8')+'_'+str(int(loss))+'.bmp')
        cv2.imwrite(path_, im)

        #tmp
        contours_path_ = os.path.join(error_path+'_better', filename.strip().decode('utf-8')+'_'+str(int(loss))+'_contour.bmp')
        cv2.imwrite(contours_path_, c_im_show)


    return loss

class Ellip_acc(object):
    
    def __init__(self):
        pass
        #self.pickle_path = cfgs.shelter_pickle_path

        #with open(self.pickle_path, 'rb') as f:
            #self.shelter_map = pickle.load(f)
    
    def ellip_loss(self, pred_ellip, gt_ellip):

        #loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2)) / (sz_[0]*sz_[1])
        loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2)) / (gt_ellip[2]*gt_ellip[3]) * 1000
 
        return loss

    def haus_loss(self, pred_ellip, gt_ellip):
        
        loss = cal_haus_loss(pred_ellip, gt_ellip)

        return loss

    #generate ellipse to compare ellispes info
    def caculate_ellip_accu_once(self, im, filename, pred, pred_pro, gt_ellip, if_valid=False):
        #gt_ellipse [(x,y), w, h]
        fn = filename.strip().decode('utf-8')
        pts = []
        _, p, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #tmp
        c_im_show = np.zeros((sz[0], sz[1], 3))
        cv2.drawContours(c_im_show, p, -1, (0, 255, 0), 1)

        
        for i in range(len(p)):
            for j in range(len(p[i])):
                pts.append(p[i][j])
        pts_ = np.array(pts)
        if pts_.shape[0] > 5:
            ellipse_info = cv2.fitEllipse(pts_)
            pred_ellip = np.array([ellipse_info[0][0], ellipse_info[0][1], ellipse_info[1][0], ellipse_info[1][1]])
            ellipse_info = (tuple(np.array([ellipse_info[0][0], ellipse_info[0][1]])), tuple(np.array([ellipse_info[1][0], ellipse_info[1][1]])), 0)
            loss = self.haus_loss(pred_ellip, gt_ellip)
        else:
            pred_ellip = np.array([0,0,0,0])
            ellipse_info = (tuple(np.array([0,0])), tuple(np.array([0,0])), 0)
            loss = self.ellip_loss(pred_ellip, gt_ellip)
        
        #sz_ = im.shape
        #loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2)) / (sz_[0]*sz_[1])
        #save worse result
        if if_valid:
            error_path = cfgs.error_path+'_valid'
        else:
            error_path = cfgs.error_path

        if fn in self.shelter_map:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.ellipse(im,ellipse_info,(0,255,0),1)
            gt_ellip_info = (tuple(np.array([gt_ellip[0], gt_ellip[1]])), tuple(np.array([gt_ellip[2], gt_ellip[3]])), 0)
            cv2.ellipse(im,gt_ellip_info,(0,0,255),1)
            path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'.bmp')
            cv2.imwrite(path_, im)

            #heatmap
            heat_map = density_heatmap(pred_pro[:, :, 1])
            cv2.imwrite(os.path.join(error_path, filename.strip().decode('utf-8')+'_heatseq_.bmp'), heat_map)
            
            #tmp
            contours_path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'_contour.bmp')
            cv2.imwrite(contours_path_, c_im_show)

        return loss

    def divide_shelter_once(self, im, filename, pred, pred_pro, gt_ellip, if_valid=False, is_save=True):
        '''
        divide the shelter and not shelter
        '''

        pts = []
        '''
        _, p, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #tmp draw contours
        sz = im.shape
        c_im_show = np.zeros((sz[0], sz[1], 3))
        cv2.drawContours(c_im_show, p, -1, (0, 255, 0), 1)
        for i in range(len(p)):
            for j in range(len(p[i])):
                pts.append(p[i][j])

        '''
        sz = im.shape
        for ii in range(sz[0]):
            for jj in range(sz[1]):
                if pred[ii][jj] > 0:
                    pts.append([jj, ii])
                    #im[ii][jj] = 255

        pts_ = np.array(pts)
        if pts_.shape[0] > 5:
            ellipse_info = cv2.fitEllipse(pts_)
            ellipse_info_ = ellipse_info
            #pred_ellip = np.array([ellipse_info[0][0], ellipse_info[0][1], ellipse_info[1][0], ellipse_info[1][1]])
            if ellipse_info[2] > 150 or ellipse_info[2] < 30:
                angle = 180
                pred_ellip = np.array([ellipse_info[0][0], ellipse_info[0][1], ellipse_info[1][0], ellipse_info[1][1]])
            else:
                angle = 90
                pred_ellip = np.array([ellipse_info[0][0], ellipse_info[0][1], ellipse_info[1][1], ellipse_info[1][0]])

            ellipse_info = (tuple(np.array([ellipse_info[0][0], ellipse_info[0][1]])), tuple(np.array([ellipse_info[1][0], ellipse_info[1][1]])), angle)
            loss = self.haus_loss(pred_ellip, gt_ellip)
        else:
            pred_ellip = np.array([0,0,0,0])
            ellipse_info = (tuple(np.array([0,0])), tuple(np.array([0,0])), 0)
            loss = self.ellip_loss(pred_ellip, gt_ellip)
        
        #loss = np.sum(np.power((np.array(gt_ellip)-pred_ellip), 2)) / (gt_ellip[2]*gt_ellip[3]) * 1000
        if is_save:
            fn = filename.strip().decode('utf-8')
            fn_full = '%s%s.bmp' % ('img', fn.split('img')[1])
            im_full = cv2.imread(os.path.join(cfgs.full_im_path, fn_full))
            im = cv2.resize(im_full, (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)


            #save worse result
            error_path = cfgs.error_path
            #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.ellipse(im,ellipse_info,(0,255,0),1)
            gt_ellip_info = (tuple(np.array([gt_ellip[0], gt_ellip[1]])), tuple(np.array([gt_ellip[2], gt_ellip[3]])), 0)
            cv2.ellipse(im,gt_ellip_info,(0,0,255),1)
            '''
            print('filename: ', fn, ' loss: ',  loss)
            print('gt_ellipse_info: ', gt_ellip)
            print('pred_ellipse_info: ', pred_ellip)
            print('gt_ellipse_info: ', gt_ellip_info)
            print('pred_ellipse_info: ', ellipse_info_)
            print('gt_ellipse_info: ', gt_ellip_info)
            print('pred_ellipse_info: ', ellipse_info)
            '''
            

       
            if loss > cfgs.loss_thresh:
                if loss > 500:
                    loss = 500
                path_ = os.path.join(error_path, filename.strip().decode('utf-8')+'_'+str(int(loss))+'.bmp')
                cv2.imwrite(path_, im)
            else:
                path_ = os.path.join(error_path+'_better', filename.strip().decode('utf-8')+'_'+str(int(loss))+'.bmp')
                cv2.imwrite(path_, im)

        return loss



    def caculate_ellip_accu(self, im, filenames, pred, pred_pro, gt_ellip, if_valid=False, is_save=True):
        sz_ = pred.shape
        loss = 0
       
        for i in range(sz_[0]):
            #loss += caculate_ellip_accu_once(im[i], filenames[i], pred[i].astype(np.uint8), pred_pro[i], gt_ellip[i], if_valid)
            loss += self.divide_shelter_once(im[i], filenames[i], pred[i].astype(np.uint8), pred_pro[i], gt_ellip[i], if_valid, is_save)

        loss = loss / sz_[0]

        return loss

def main():
    pred_ellip1 = np.array([20, 30, 20, 40])
    pred_ellip2 = np.array([19, 30, 20, 40])
    gt_ellip = np.array([21, 30, 20, 40])
    e_acc = Ellip_acc()
    dis1 = e_acc.haus_loss(pred_ellip1, gt_ellip)
    dis2 = e_acc.haus_loss(pred_ellip2, gt_ellip)
    print(dis1)
    print(dis2)

if __name__ == '__main__':
    main()
