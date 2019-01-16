import numpy as np
import os
import pdb
from six.moves import cPickle as pickle
import random

try:
    from .cfgs.config_train_u_net import cfgs 
except Exception:
    from cfgs.config_train_u_net import cfgs


ratio = [cfgs.IMAGE_SIZE[0]/1080, cfgs.IMAGE_SIZE[1]/1920]


def my_read_dataset(data_dir, anno_dir):
    pickle_filename = cfgs.data_pickle_fn
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = read_txt_data(os.path.join(data_dir), anno_dir)
        print ("Pickling ...")
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

def _rect(gt):
    
    lx=float(gt[1])
    ly=float(gt[2])
    rx=float(gt[3])
    ry=float(gt[4])
    w=rx-lx
    h=ry-ly
    cx=(lx+rx)/2*ratio[1]
    cy=(ly+ry)/2*ratio[0]
    ellip_info = [cx, cy, w*ratio[0], h*ratio[1]]
    
    return ellip_info

def _rect_s7(gt):
    
    lx=float(gt[1])
    ly=float(gt[2])
    rx=float(gt[3])
    ry=float(gt[4])
    w=rx-lx
    h=ry-ly
    cx=(lx+rx)/2*ratio[1]
    cy=(ly+ry)/2*ratio[0]
    ellip_info = [cx, cy, w*ratio[1], h*ratio[0]]
    
    return ellip_info



def read_txt_data(folder_dir, anno_folder):
    
    if not os.path.exists(folder_dir):
        raise Exception('txt file folder %s not found' % folder_dir)
    
    seq_list = {}
    #radio = [224/1920, 224/1080, 112/1920, 112/1080]
    for txt_file in cfgs.file_list:
        path_ = os.path.join(folder_dir, txt_file)
        if not os.path.exists(path_):
            raise Exception('No file %s found' % txt_file)
        if not os.path.exists(anno_folder):
            raise Exception('No file %s found' % anno_path)
        print(txt_file)
        if 'train' in txt_file:
            seq_list['training'] = []
            directory = 'training'
        elif 'validation' in txt_file:
            seq_list['validation'] = []
            directory = 'validation'
        else:
            raise Exception('txt file does not match')
          
        f = open(path_, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split('*')

            seq_set = []
            #1. get path of sequence, if not sequence, seq_num=0
            for i in range(1, cfgs.seq_num+1):
                seq_set.append(line[i].split(' ')[0])
            #2. get path of current frame
            line_cur_s = line[cfgs.seq_num+1].split(' ')
            cur_frame = line_cur_s[0]
            cur_frame_s = cur_frame.split("/")
            #3. get filename
            filename = '%s%s' % (cur_frame_s[len(cur_frame_s)-3], os.path.splitext(cur_frame_s[-1])[0])
            cur_filename = '%s%s' % (cur_frame_s[len(cur_frame_s)-3], os.path.splitext(cur_frame_s[-1])[0])

            #cur_filename = '%s' % (os.path.splitext(cur_frame_s[-1])[0])

            if 's7' in filename or 's3' in filename or 'simg' in filename:
                ellip_info = _rect_s7(line_cur_s)
            else:
                ellip_info = _rect(line_cur_s)
            #4. get path of annotation frame
            cur_frame = os.path.join(cfgs.im_folder, cur_filename+'.bmp')
            anno_frame = os.path.join(anno_folder, filename+'.bmp')
            print('fn: %s' % filename)
            print('cur_frame: %s' % cur_frame)
            print('anno_frame: %s' % anno_frame)
            #5. generate record
            if os.path.exists(anno_frame) and os.path.exists(cur_frame):
                record = {'current':cur_frame, 'seq':seq_set, 'annotation':anno_frame, 'ellip_info':ellip_info, 'filename':filename}
              
                seq_list[directory].append(record)
            else:
                print("Annotation file or image file not found for %s - Skipping" % filename)

            
        #random.shuffle(seq_list[directory])
        no_of_images = len(seq_list[directory])
        print('No of %s files %d' % (directory, no_of_images))

    return seq_list


