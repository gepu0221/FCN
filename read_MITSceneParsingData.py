__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'



#data_dir:Data_zoo/MIT_SceneParsing
def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print ("Pickling ...")
        #data_dir:Data_zoo/MIT_SceneParsing/pickle_filepath
        with open(pickle_filepath, 'wb') as f:
            #序列化对象
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    #return the training_records path list and validatation_records path list
    return training_records, validation_records

def my_read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = create_image_lists(os.path.join(data_dir))
        print ("Pickling ...")
        #data_dir:Data_zoo/MIT_SceneParsing/pickle_filepath
        with open(pickle_filepath, 'wb') as f:
            #序列化对象
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    #return the training_records path list and validatation_records path list
    return training_records, validation_records

#use the train model to test whole video data
def read_validation_video_data(data_dir):
    pickle_filename = "Valid_video.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = create_image_lists_video(os.path.join(data_dir))
        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file!")
    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        validation_video_records = result['valid_video']
        del result

    return validation_video_records 
    

def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        #file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'bmp')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                #分离文件名与扩展名；默认返回(fname,fextension)元组
                filename = os.path.splitext(f.split("/")[-1])[0]
                #annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.bmp')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list


def create_image_lists_video(image_dir):
    if not gfile.Exists(image_dir):
        print("Image dirctory '%s' not found." % image_dir)
        return None
    image_list = {}
    #save the path of the images
    file_list = []
        
    directory = 'valid_video'
    image_list[directory] = []
    directory = 'valid_video'
    #each image path
    file_glob = os.path.join(image_dir, "images", directory, '*' + 'bmp')
    #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    file_list.extend(glob.glob(file_glob))

    if not file_list:
        print('No files found')
    else:
        for f in file_list:
            filename = os.path.splitext(f.split("/")[-1])[0]
            record = {'image' : f, 'filename': filename}
            image_list[directory].append(record)

    random.shuffle(image_list[directory])
    no_of_images = len(image_list[directory])
    print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list

            

#update by gp
#change by new net which output label
def read_dataset_new_net(data_dir,txt_data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        data_dir_array=txt_data_dir.split('*')
        print(data_dir_array)
        result = create_image_gt(data_dir_array)
        print ("Pickling ...")
        #data_dir:Data_zoo/MIT_SceneParsing/pickle_filepath
        with open(pickle_filepath, 'wb') as f:
            #序列化对象
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result
    
    #return the training_records path list and validatation_records path list
    return training_records, validation_records

def create_image_gt(annotation_dir):
    #annotation_dir=['training.txt','validation.txt']
    directory_name=['training','validation']
    image_gt_list={}
    for n in range(len(annotation_dir)):
        image_gt_list[directory_name[n]] = []
        if not os.path.exists(annotation_dir[n]):
            print('annotation file',annotation_dir[n],'is not found')
            return None
        
        print('annotation_dir',annotation_dir[n])
        print('directory_name',directory_name[n])
        gt_file=open(annotation_dir[n])
    
        gt=[]
        while True:
            line=gt_file.readline()
            if not line:
                break
            line=line.split(' ')
            gt.append(line)
    
        gt_file.close()
        l=len(gt)
    
        for i in range(l):
            image_file=gt[i][0]
            gt_image_file=gt[i][1]
            if os.path.exists(image_file) and os.path.exists(gt_image_file):
                #[center.x,center.y,width,height,angle]
                #anno_data=np.array([gt[i][2],gt[i][3],gt[i][4],gt[i][5],gt[i][6]])
                anno_data=(gt[i][2],gt[i][3],gt[i][4],gt[i][5],0)
                print('image',image_file,'annotation',gt_image_file,'annotation_data',anno_data)
                record={'image':image_file,'annotation':gt_image_file,'annotation_data':anno_data}
               
                image_gt_list[directory_name[n]].append(record)
            else:
                print("Annotation file not found for %s -Skipping"%filename)
                
        random.shuffle(image_gt_list[directory_name[n]])
        num_of_images=len(image_gt_list[directory_name[n]])
        print('No. of %s files:%d' %(directory_name[n],num_of_images))
        
    return image_gt_list
        
    
        
