import os
import numpy as np
import argparse
import random
from cfgs.config import cfg

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_file_dir',help='the path to load data txt')
    parser.add_argument('--save_file_dir',help='the path to save result')
    parser.add_argument('--small_data_size',help='the small train data size')
    args=parser.parse_args()
    
    save_folder=args.save_file_dir
    
    data_folder=args.load_file_dir
    data_folder_list=[f for f in os.listdir(data_folder)]
    f_num=np.size(data_folder_list)
    records=[]
    
    for j in range(f_num):
        sub_data_folder=os.path.join(data_folder,data_folder_list[j])
        data_txt_list=[f for f in os.listdir(sub_data_folder) if f.endswith("groundtruth_copy0504.txt")]
        d_num=np.size(data_txt_list)
        #print(data_txt_list)
        print(d_num)
    
        
        for i in range(d_num):
            data_path=os.path.join(sub_data_folder,data_txt_list[i])
            #count=0
            file=open(data_path)
            while True:
                line=file.readline()
                #line=line+'0'
                print(line)
                if not line:
                    break
                
                records.append(line)
        
            file.close()
    
    random.shuffle(records)
    print("total size:",len(records))
  
    total_num = len(records)
    test_num = int(cfg.test_ratio*total_num)
    #test_num=120
    train_num = total_num - test_num
    train_records = records[0:train_num]
    #Get small size train_data
    train_records = train_records[0:int(args.small_data_size)]
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
    
    
