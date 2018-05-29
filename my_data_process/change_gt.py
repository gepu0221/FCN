import tensorflow as tf
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #the path to load data
    parser.add_argument('--load_train_data',help='use to load the data,the defalut path is ~/repos/data/train_data')
    #use to label the folder name(xxxxx) in each train data folder(eg.student(5)/xxxx/img00000.bmp)
    parser.add_argument('--train_image_folder', default='Image',
                        help='use to label the folder name(xxxxx)in each train data folder(eg.student(5)/xxxx/img00000.bmp)')
    parser.add_argument('--img_type',default='bmp',help='the image type of train data')
    args = parser.parse_args()
    train_data_folder=args.load_train_data
    train_image_folder=args.train_image_folder
    img_type=args.img_type
    #train_data_folder=os.path.join() 
    videos_list=[v for v in os.listdir(train_data_folder)] 
    num_v=len(videos_list)
    for i in range(num_v):
        video_folder=os.path.join(train_data_folder,videos_list[i])
        print(video_folder)
        gt_file=os.path.join(video_folder,'out1.txt')
        gt=np.genfromtxt(gt_file,delimiter=' ')
        num=int(np.size(gt)/5)
        print(num)
    
        new_gt=os.path.join(video_folder,'groundtruth_copy0516.txt')
        img_path=os.path.join(video_folder,train_image_folder)
        f=open(new_gt,'w')
        for j in range(num):
            gt_=gt[j]
            img_name='img'+"%05d"%gt_[0]+'.'+img_type
            img_name=os.path.join(img_path,img_name)
            #str_=img_name+' '+str(gt_[1]-gt_[3]/2)+' '+str(gt_[2]-gt_[4]/2)+' '+str(gt_[3])+' '+str(gt_[4])
            #str_=img_name+' '+str(gt_[2]-gt_[4]/2)+' '+str(gt_[1]-gt_[3]/2)+' '+str(gt_[4])+' '+str(gt_[3])+' '+str(0)
            #(xmin,ymin,xmax,ymax)
            str_=img_name+' '+str(gt_[2]-gt_[4]/2)+' '+str(gt_[1]-gt_[3]/2)+' '+str(gt_[2]+gt_[4]/2)+' '+str(gt_[1]+gt_[3]/2)+' '+str(0)
            f.write(str_+'\n')
        f.close()
    
        print('end')

        
        
    
