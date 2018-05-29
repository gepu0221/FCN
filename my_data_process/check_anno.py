#The code is used to check the generation annotions if label correctly.(Created by gp 20180505 18:49)
import tensorflow as tf
import cv2
import os
import argparse
import numpy as np
from EFCHandler import EFCHandler
from ellipse_my import ellipse_my

    
root_dataset='/home/gp/repos/data'
label_data_folder='fcn_anno'
image_data_folder='train_data'
image_save_folder='image_save'
file_name=['training','validation']
txt_file_name=['validation.txt','training.txt']
extend_rate=1.2
resize_crop_sz=224
part_num=200
coef_num=30
#the round pixel to get
step=3
    
if not gfile.Exists(image_dir):
 67         print("Image directory '" + image_dir + "' not found.")
  68         return None
   69     directories = ['training', 'validation']
    70     image_list = {}
     71 
      72     for directory in directories:
       73         file_list = []
        74         image_list[directory] = []
         75         #file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
          76         file_glob = os.path.join(image_dir, "images", directory, '*.' + 'bmp')
           77         file_list.extend(glob.glob(file_glob))
            78 
             79         if not file_list:
              80             print('No files found')
               81         else:
                82             for f in file_list:
                 83                 filename = os.path.splitext(f.split("/")[-1])[0]
                  84                 #annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                   85                 annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.bmp')
                    86                 if os.path.exists(annotation_file):
                     87                     record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                      88                     image_list[directory].append(record)
                       89                 else:
                        90                     print("Annotation file not found for %s - Skipping" % filename)
                         91 
                          92         random.shuffle(image_list[directory])
                           93         no_of_images = len(image_list[directory])
                            94         print ('No. of %s files: %d' % (directory, no_of_images))
def check_anno_region(root_dataset,image_data_folder):
    data_path=os.path.join(root_dataset,image_data_folder)
    if not gfile.Exists(data_path):
        print("Image directory *%s* not found." % data_path)
        return None
   directores = ['training', 'validation']

   for directory in directories:
        file_list = []
        file_glob = os.path

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',help = 'the root path of image data needed to check annotations')
    parser.add_argument('--image_folder',help = 'the folder of the image' )
    args = parser.parse_args()
