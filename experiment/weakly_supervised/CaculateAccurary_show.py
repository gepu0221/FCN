import numpy as np
import cv2
import time 
import pdb
import os
from ellipse_my import ellipse_my
from generate_heatmap import density_heatmap
from six.moves import cPickle as pickle
from Hausdorff_dis import cal_haus_loss


try:
    from .cfgs.config_acc import cfgs
except Exception:
    from cfgs.config_acc import cfgs

def get_nine_half(cx,cy,x,y):
    p_i_x=int((x*9+cx)/10)
    p_i_y=int((y*9+cy)/10)
    p_o_x=int(2*x-p_i_x)
    p_o_y=int(2*y-p_i_y)
    
    return p_i_x,p_i_y,p_o_x,p_o_y

#Create ellipse error related folder.
def create_ellipse_f():
    train_error_path = cfgs.error_path
    valid_error_path = cfgs.error_path+'_better'
    if not os.path.exists(train_error_path):
        os.makedirs(train_error_path)
    if not os.path.exists(valid_error_path):
        os.makedirs(valid_error_path)


def remove_small_area(im):
    
    im_ = im.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(im_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_n = []
    for i in range(len(contours)):
        num = cv2.contourArea(contours[i])
        if num < cfgs.small_area_num:
            contours_n.append(contours[i])
    cv2.drawContours(im, contours_n, -1, (0, 0, 0), -1)
 
    return im



def save_max_area(im):
    
    im_ = im.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(im_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_n = []
    max_num = 0
    max_idx = 0
    for i in range(len(contours)):
        num = cv2.contourArea(contours[i])
        print('idx: %d, num: %g' % (i, num))
        if num > max_num:
            max_idx = i
            max_num = num
    print('max_idx: %d, num: %g' % (max_idx, max_num)) 
    contours_n.append(contours[max_idx])
    cv2.drawContours(im, contours_n, -1, (0, 0, 0), -1)
 
    return im



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
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        #tmp       
        fn = filename.strip().decode('utf-8')
        #fn_full = '%s%s.bmp' % ('img', fn.split('img')[1])
        #im_full = cv2.imread(os.path.join(cfgs.full_im_path, fn_full))
        #im = cv2.resize(im_full, (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)

        fn_full = '%s%s%s.bmp' % ('s8', 'img', fn.split('img')[1])
        fn_full_path = os.path.join(cfgs.full_im_path, fn_full)
        im_full = cv2.imread(fn_full_path)
        im = cv2.resize(im_full, (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)
        

        
        #pred = remove_small_area(pred)
        for ii in range(sz[0]):
            for jj in range(sz[1]):
                #if pred[ii][jj] > 0:
                if pred[ii][jj] == 2:
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
            #loss = self.haus_loss(pred_ellip, gt_ellip)
        else:
            pred_ellip = np.array([0,0,0,0])
            ellipse_info = (tuple(np.array([0,0])), tuple(np.array([0,0])), 0)
            #loss = self.ellip_loss(pred_ellip, gt_ellip)
        loss = 0
        if is_save:
            
            #save worse result
            error_path = cfgs.error_path
            #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.ellipse(im,ellipse_info,(0,255,0),1)
            #gt_ellip_info = (tuple(np.array([gt_ellip[0], gt_ellip[1]])), tuple(np.array([gt_ellip[2], gt_ellip[3]])), 0)
            #cv2.ellipse(im,gt_ellip_info,(0,0,255),1)
            part_label=ellipse_my(ellipse_info)
            c_x=ellipse_info[0][0]
            c_y=ellipse_info[0][1]
    
    
            for i in range(len(part_label)):
                x=int(part_label[i][1])
                y=int(part_label[i][0])
                if y<len(im) and x<len(im[0]):
                    im[y][x][0]=0
                    im[y][x][1]=255
                    im[y][x][2]=0
                    p_i_x,p_i_y,p_o_x,p_o_y=get_nine_half(c_x,c_y,x,y)
                    #p_i_y,p_i_x,p_o_y,p_o_x=get_nine_half(c_x,c_y,x,y)
                    cv2.line(im,(p_i_x,p_i_y),(p_o_x,p_o_y),(0,255,0),1)
            

       
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
