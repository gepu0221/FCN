import numpy as np

#data_format:[image_path center.x center.y width height angle]

import os
import numpy as np
import argparse
import random
from cfgs.config import cfg

def _rect(gt):
    
    lx=int(float(gt[1]))
    ly=int(float(gt[2]))
    rx=int(float(gt[3]))
    ry=int(float(gt[4]))
    w=rx-lx
    h=ry-ly
    
    return lx,ly,rx,ry,w,h

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_file_dir',help='the path to load data txt')
    parser.add_argument('--save_file_dir',help='the path to save result')
    args=parser.parse_args()
    
    save_folder=args.save_file_dir
    
    data_folder=args.load_file_dir
    data_folder_list=[f for f in os.listdir(data_folder)]
    f_num=np.size(data_folder_list)
    records=[]
    
    for j in range(f_num):
        sub_data_folder=os.path.join(data_folder,data_folder_list[j])
        data_txt_list=[f for f in os.listdir(sub_data_folder) if f.endswith("groundtruth.txt")]
        d_num=np.size(data_txt_list)
        #print(data_txt_list)
        print(d_num)
    
        
        for i in range(d_num):
            data_path=os.path.join(sub_data_folder,data_txt_list[i])
            #count=0
            file=open(data_path)
            while True:
                line=file.readline()
                if not line:
                    break
                gt=line.split(' ')
                image_split=gt[0].split('/')
                image_path=
                
                lx,ly,rx,ry,w,h=_rect(gt)
                #print('lx:',lx,'ly:',ly,'rx',rx,'ry',ry,'w',w,'h',h)
                cx=lx+w/2
                cy=ly+h/2
                line=gt[0]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+' '+gt[5]
                print(line)
                records.append(line)
        
            file.close()
    
    random.shuffle(records)
    print("total size:",len(records))
  
    total_num = len(records)
    test_num = int(cfg.test_ratio*total_num)
    train_num = total_num - test_num
    train_records = records[0:train_num]
    tests_records = records[train_num:]
    
    train_out_file_path=os.path.join(save_folder,cfg.train_list[0])
    train_out_file = open(train_out_file_path,'w')
    for record in train_records:
        train_out_file.write(record)
    train_out_file.close()

    test_out_file_path=os.path.join(save_folder,cfg.test_list)
    test_out_file = open(test_out_file_path,'w')
    for record in tests_records:
        test_out_file.write(record)
    test_out_file.close()
    
    