#This code is use the mothod of drawing ellipse to genereate soft targets.
import tensorflow as tf
import cv2
import os
import time
import numpy as np
from EFCHandler import EFCHandler
from ellipse_my import ellipse_my
from polar_trans import polar_transform_pixel
    
root_dataset='/home/gp/repos/data'
label_data_folder='fcn_anno'
image_data_folder='train_data'
image_save_folder='image_save20180529_s6_soft'
extend_rate=1.2
resize_crop_sz=225
part_num=200
coef_num=30
#the round pixel to get
step=3
#ues to generate soft targets
expand_num=254

def _rect(gt):
    
    lx=int(float(gt[1]))
    ly=int(float(gt[2]))
    rx=int(float(gt[3]))
    ry=int(float(gt[4]))
    w=rx-lx
    h=ry-ly
    
    return lx,ly,rx,ry,w,h

def get_point(x,y):
    s_x=int(x-step)
    e_x=int(x+step+step)
    s_y=int(y-step)
    e_y=int(y+step+step)
    
    if step==0:
        e_x=int(e_x+1)
        e_y=int(e_y+1)
    
    return s_x,e_x,s_y,e_y

def get_card_angle(p_angle):

    c_angle = p_angle / (2*np.pi) *360

    return c_angle

label_folder=os.path.join(root_dataset,label_data_folder)
file_list=[f for f in os.listdir(label_folder) if f.endswith(".txt")]
t=0
for f in file_list:
    
    print('txt_name',f)
    print('----------------------------')
    if f.find("test")>0:
        file_name = 'validation'
    else:
        file_name = 'training'
    print('filename', file_name)    
    records=[]
    f_path=os.path.join(root_dataset,label_data_folder,f)
    gt_file=open(f_path)
    
    gt=[]
    while True:
        line=gt_file.readline()
        if not line:
            break
        line=line.split(' ')
        gt.append(line)
    
    gt_file.close()
    l=len(gt)
    print('The number of data is %d' % l)
    
    folder=os.path.join(root_dataset,image_save_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    img_folder0=os.path.join(folder,'images')
    if not os.path.exists(img_folder0):
        os.mkdir(img_folder0)
    img_folder1=os.path.join(img_folder0,file_name)
    if not os.path.exists(img_folder1):
        os.mkdir(img_folder1)
        
    gt_folder0=os.path.join(folder,'annotations')
    if not os.path.exists(gt_folder0):
        os.mkdir(gt_folder0)
    gt_folder1=os.path.join(gt_folder0,file_name)
    if not os.path.exists(gt_folder1):
        os.mkdir(gt_folder1)
    
    size=(resize_crop_sz,resize_crop_sz)
    
    
    for i in range(l):
        print('gt[i][0]',gt[i][0])
        im = cv2.imread(gt[i][0],cv2.IMREAD_COLOR)
        frame_sz = np.array([len(im),len(im[0])])
        draw_im=np.zeros((frame_sz[0],frame_sz[1],3))
        gt_im0 = np.zeros((frame_sz[0],frame_sz[1],1))
        gt_im1 = np.zeros((frame_sz[0],frame_sz[1],1))
        gt_im2 = np.ones((frame_sz[0],frame_sz[1],1))*255
        lx,ly,rx,ry,w,h=_rect(gt[i])
        cx=lx+w/2
        cy=ly+h/2
       
        img_name_split=gt[i][0].split('/')
        str_l=len(img_name_split)
        img_name=os.path.join(img_folder1,img_name_split[str_l-3]+img_name_split[str_l-1])
       
        crops=cv2.resize(im,size,interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_name,crops)
        
        count = 2
        h_ = h
        w_ = w
        h_s = h
        w_s = w
        rate_pixel = 255
        center = (cx, cy)
        angle_ = 0
        for s in range(step):

           axis_ = (h_, w_)
           ellipse_info=(center, axis_, angle_)
           cv2.ellipse(draw_im, ellipse_info, (0,rate_pixel,0),1)
           if h_s>0 and w_s>0:
               axis_s = (h_s,w_s)
               ellipse_info_s = (center, axis_s, angle_)
               cv2.ellipse(draw_im, ellipse_info_s, (0,rate_pixel,0),1)
           h_ += count
           w_ += count
           h_s -= count
           w_s -= count
        rate_pixel -=1
        for t in range(expand_num):
           axis_ = (h_, w_)
           ellipse_info=(center, axis_, angle_)
           cv2.ellipse(draw_im, ellipse_info, (0,rate_pixel,0),1)
           if h_s>0 and w_s>0:
               axis_s = (h_s,w_s)
               ellipse_info_s = (center, axis_s, angle_)
               cv2.ellipse(draw_im, ellipse_info_s, (0,rate_pixel,0),1)
           rate_pixel -= 1
           h_ += count
           w_ += count
           h_s -= count
           w_s -= count
        gt_im1 = gt_im1 + np.expand_dims(draw_im[:,:,1], axis=2)
        gt_im2 = gt_im2 - gt_im1
        gt_im = np.concatenate((gt_im0, gt_im1, gt_im2), axis=2)
        gt_img_name=os.path.join(gt_folder1,img_name_split[str_l-3]+img_name_split[str_l-1])
        gt_crops=cv2.resize(gt_im,size,interpolation=cv2.INTER_AREA)
        #cv2.imwrite(gt_img_name,draw_im)
        cv2.imwrite(gt_img_name, gt_crops)
        if i % 1000 == 0:
            print('Now is %d...' % i)
       
  
