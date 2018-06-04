#This code is used to saperate the training data into training and validation,two parts.
#So the data is transformed from training,validation to training,validation,test.
import numpy as np
import glob
import os
import shutil

root_dir="/home/gp/repos/FCN"
folder_dir="image_save20180530_soft_total_valid"
#folder_dir="image_test_valid"
sub_folder_dir=['images', 'annotations']
directory='training'
move_dire='validation'
valid_rate=0.2

file_list = []
im_path = os.path.join(root_dir, folder_dir, sub_folder_dir[0], directory)
anno_path = os.path.join(root_dir, folder_dir, sub_folder_dir[1], directory)
im_valid_path = os.path.join(root_dir, folder_dir, sub_folder_dir[0], move_dire)
anno_valid_path = os.path.join(root_dir, folder_dir, sub_folder_dir[1], move_dire)
print('im_path: ',im_path)

file_glob = os.path.join(im_path, '*.' + 'bmp')
file_list.extend(glob.glob(file_glob))

if not file_list:
    print('No files found')
else:
    valid_num_ = int(len(file_list)*valid_rate)
    print('valid_num:%d' % valid_num_)
    count=0
    perm = np.arange(len(file_list))
    np.random.shuffle(perm)
    for i in range(valid_num_):
        f = file_list[perm[i]]
        filename = os.path.splitext(f.split("/")[-1])[0]
        annotation_file = os.path.join(anno_path, filename+ '.bmp')
        dst_annotation_file = os.path.join(anno_valid_path, filename+'.bmp')
        dst_image_file = os.path.join(im_valid_path, filename+'.bmp')
        if os.path.exists(annotation_file):
            print(f)
            print(annotation_file)
            print('-----------------------------------')
            shutil.move(f, dst_image_file)
            shutil.move(annotation_file, dst_annotation_file)

            

