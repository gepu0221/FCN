import numpy as np
from ellipse_my import ellipse_my
import cv2
import time

def generate_heat_map(sz_,pred_value):
    #im_h=np.zeros((sz_[0],sz_[1],1))
    for i in range(sz_[0]):
        for j in range(sz_[1]):
            pred_value[i][j][0]=int(pred_value[i][j][0]*255)
    print('pred_value',pred_value)
            
    return pred_value
            
def get_nine_half(cx,cy,x,y):
    p_i_x=int((x*9+cx)/10)
    p_i_y=int((y*9+cy)/10)
    p_o_x=int(2*x-p_i_x)
    p_o_y=int(2*y-p_i_y)
    
    return p_i_x,p_i_y,p_o_x,p_o_y

#Use to check the prediction segmentation result.
#label with green.
def pred_visualize(image,pred):
    
    i_sz=image.shape
    for i in range(i_sz[0]):
        for j in range(i_sz[1]):
            if pred[i][j]!=0:
                image[i,j,0]=0
                image[i,j,1]=255
                image[i,j,2]=0
                
    #return image,pred
    return image

#Use to check the annotaion(true) segmentation result.
#label with blue
def anno_visualize(image,anno):
    
    i_sz=image.shape
    for i in range(i_sz[0]):
        for j in range(i_sz[1]):
            if anno[i][j]!=0:
                image[i,j,0]=0
                image[i,j,1]=0
                image[i,j,2]=255
                
    #return image,pred
    return image


def fit_ellipse(image,pred):
    #all 69.51ms
    
    i_sz=pred.shape
    pts=[]
    
    #13.4ms/40.75ms
    for i in range(i_sz[0]):
        for j in range(i_sz[1]):
            if pred[i][j]!=0:
                pts.append([j,i])
                
    p=np.array(pts)
    #print('p',p)            
    #3.43ms/54.23ms
    ellipse_info_=cv2.fitEllipse(p)
    if ellipse_info_[2]>90:
        angle_=180
    else:
        angle_=0
    ellipse_info=(ellipse_info_[0],ellipse_info_[1],angle_)
    cv2.ellipse(image,ellipse_info,(0,255,0),1)
    #print(ellipse_info)
    
    #11.8ms/57.67ms
    #***********************
    part_label=ellipse_my(ellipse_info)
    c_x=ellipse_info[0][0]
    c_y=ellipse_info[0][1]
    
    
    for i in range(len(part_label)):
        x=int(part_label[i][1])
        y=int(part_label[i][0])
        if y<len(image) and x<len(image[0]):
            image[y][x][0]=0
            image[y][x][1]=255
            image[y][x][2]=0
            p_i_x,p_i_y,p_o_x,p_o_y=get_nine_half(c_x,c_y,x,y)
            #p_i_y,p_i_x,p_o_y,p_o_x=get_nine_half(c_x,c_y,x,y)
            cv2.line(image,(p_i_x,p_i_y),(p_o_x,p_o_y),(0,255,0),1)
    #**********************
    size=(224,224)
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        
    return image

def fit_ellipse_findContours(image,pred):
    #all 69.51ms
    
    i_sz=pred.shape
    pts=[]
    _,p,hierarchy = cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    for i in range(len(p)):
        for j in range(len(p[i])):
            pts.append(p[i][j])
                
    pts_=np.array(pts)
    #3.43ms/54.23ms
    ellipse_info_=cv2.fitEllipse(pts_)
    if ellipse_info_[2]>90:
        angle_=180
    else:
        angle_=0
    ellipse_info=(ellipse_info_[0],ellipse_info_[1],angle_)
    cv2.ellipse(image,ellipse_info,(0,255,0),1)
    #print(ellipse_info)
    
    #11.8ms/57.67ms
    #***********************
    part_label=ellipse_my(ellipse_info)
    c_x=ellipse_info[0][0]
    c_y=ellipse_info[0][1]
    
    
    for i in range(len(part_label)):
        x=int(part_label[i][1])
        y=int(part_label[i][0])
        if y<len(image) and x<len(image[0]) and y >= 0 and x>=0: 
            image[y][x][0]=0
            image[y][x][1]=255
            image[y][x][2]=0
            p_i_x,p_i_y,p_o_x,p_o_y=get_nine_half(c_x,c_y,x,y)
            #p_i_y,p_i_x,p_o_y,p_o_x=get_nine_half(c_x,c_y,x,y)
            cv2.line(image,(p_i_x,p_i_y),(p_o_x,p_o_y),(0,255,0),1)
    #**********************
    size=(224,224)
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        
    return image

def fit_ellipse_train(pred_batch):
    
    p_sz=pred_batch.shape
    ellip_info_batch=np.empty((p_sz[0],5))
  
    pts=[]
    
    for i in range(i_sz[0]):
        for j in range(i_sz[1]):
            if pred[i][j]!=0:
                pts.append([j,i])
                
    for n in range(p_sz[0]):
        for i in range(p_sz[1]):
            for j in range(p_sz[2]):
                if pred[i][j]!=0:
                    pts.append([j,i])
        p=np.array(pts)
        ellipse_info=cv2.fitEllipse(p)
        #e_info<array>:[center.x,ceter.y,axis.width.axis.heigth,angle]
        e_info=np.array([ellipse_info[0][0],ellipse_info[0][1],ellipse_info[1][0],ellipse_info[1][1],ellipse_info[2]])
        
        ellip_info_batch[n]=e_info
        pts.clear()
    
    return ellip_info_batch
