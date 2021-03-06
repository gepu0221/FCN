#Multithreading batchreader for sequence FCN and add ellipse info to evaluate accuracy first modifed on 201800801.
import numpy as np 
import tensorflow as tf
import scipy.misc as misc
import os
import random
import pdb
import time
import cv2
import read_data as scene_parsing

try:
    from .cfgs.config_train_m import cfgs 
except Exception:
    from cfgs.config_train_m import cfgs
image_options = {'resize':True, 'resize_size':cfgs.IMAGE_SIZE}

#fuse current frame with sequence 
def fuse_seq(cur_filename, seq_set_filename):
    #
    if cfgs.cur_channel == 3:
        current_frame = transform_rgb(cur_filename)
    elif cfgs.cur_channel == 1:
        current_frame = transform_gray(cur_filename)
    frame = current_frame
    for i in range(0,cfgs.seq_num):
        frame = tf.concat((frame, transform_gray(seq_set_filename[i])), 2)
   
    return frame

def fuse_seq_not_cur(seq_set_filename):
 
    frame = transform_gray(seq_set_filename[0])
    for i in range(1, cfgs.seq_num):
        frame = tf.concat((frame, transform_gray(seq_set_filename[i])), 2)

    return frame

#fuse current frame with sequence mask 
def fuse_seq_mask(cur_filename, seq_set_filename):
    #
    if cfgs.cur_channel == 3:
        current_frame = transform_rgb(cur_filename)
    elif cfgs.cur_channel == 1:
        current_frame = transform_gray(cur_filename)
    frame = current_frame
    for i in range(0,cfgs.seq_num):
        frame = tf.concat((frame, tf.cast(transform_anno_test(seq_set_filename[i]), tf.float32)), 2)
   
    return frame



def transform_misc(filename):
    image = misc.imread(filename)

    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = misc.imresize(image,[resize_size, resize_size], interp='nearest')
    else:
        resize_image = image

    return np.array([resize_image])


def transform_rgb(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
        
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = tf.image.resize_bilinear([image], size=[resize_size, resize_size])[0]
    else:
        resize_image = image

    return resize_image

def transform_gray(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
        
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = tf.image.resize_bilinear([image], size=[resize_size, resize_size])[0]
    else:
        resize_image = image

    return resize_image

def transform_anno(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1)/128
    
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = tf.image.resize_bilinear([image], size=[resize_size, resize_size])[0]
    else:
        resize_image = image

    return resize_image+0.5

def transform_anno_test(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image)
    image = tf.cast(tf.expand_dims(tf.reduce_mean(image, 2),2), tf.int32)

    return image

def transform_anno_soft(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image)
    image = tf.concat((tf.expand_dims(image[:,:,0], 2), tf.expand_dims(image[:,:,1], 2)), 2)
    image = image/255

    return image


#----------Read list Begin------------

#use ellipse info to evaluate accuracy
def _read_images_list(files, shuffle=True):

    if shuffle:
        random.shuffle(files)

    #seq, cur, anno, ellip_info, filename
    seq_images = [item['seq'] for item in files]
    cur_images = [item['current'] for item in files]
    ellip_infos = [item['ellip_info'] for item in files]
    #print(len(ellip_infos))

    annotations = [item['annotation'] for item in files]
    annos_soft = [item['anno_soft'] for item in files]

    filenames = [item['filename'] for item in files]
    data = tf.data.Dataset.from_tensor_slices((seq_images, cur_images, ellip_infos, annotations, annos_soft, filenames))

    return data

def _read_video_list(files, shuffle=True):
    
    if shuffle:
        random.shuffle(files)

    #seq, cur, filename
    seq_images = [item['seq'] for item in files]
    cur_images = [item['current'] for item in files]
    filenames = [item['filename'] for item in files]

    data = tf.data.Dataset.from_tensor_slices((seq_images, cur_images, filenames))

    return data
#2. read list for result recover
def _read_images_list_re(files, shuffle=True):
    if shuffle:
        random.shuffle(files)
    
    #seq, cur, anno, ellip_info, re_recover, filename
    seq_images = [item['seq'] for item in files]
    cur_images = [item['current'] for item in files]
    ellip_infos = [item['ellip_info'] for item in files]
    re_images = [item['re_recover'] for item in files]

    annotations = [item['annotation'] for item in files]

    filenames = [item['filename'] for item in files]
    data = tf.data.Dataset.from_tensor_slices((seq_images, cur_images, ellip_infos, annotations, re_images, filenames))

    return data




#----------------Read List End----------------

#----------------Parse List Begin-------------
def _parse_record(seq_filename_set, cur_filename, ellip_info, anno_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)

    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)
    #get annotation
    anno_im = transform_anno_test(anno_filename)
    return fuse_im, current_frame, ellip_info, anno_im, filename

def _parse_record_mask(seq_filename_set, cur_filename, ellip_info,  anno_filename, filename):
    #fuse_im_seq(anot current)
    fuse_im = fuse_seq_not_cur(seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)

    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)

    anno_im = transform_anno_test(anno_filename)

    return fuse_im, current_frame, ellip_info, anno_im, filename

def _parse_record_re(seq_filename_set, cur_filename, ellip_info, anno_filename, re_filename, filename):
    '''
    Args:
    re_filename: re_recover_filename
    '''
    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)
    #get annotation
    anno_im = transform_anno_test(anno_filename)
    re_im = transform_anno_test(re_filename)
    return current_frame, ellip_info, anno_im, re_im, filename


def _parse_vis_record(seq_filename_set, cur_filename, ellip_info, anno_filename, filename):
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)
    #current frame
    cur_im = transform_rgb(cur_filename)
    #get annotation
    #anno_im = tf.cast(transform_anno(anno_filename), tf.int32)
    anno_im = transform_anno_test(anno_filename)

    return fuse_im, cur_im, ellip_info, anno_im, filename

def _parse_vis_mask_record(seq_filename_set, cur_filename, ellip_info, anno_filename, filename):
    #fuse sequence
    fuse_im = fuse_seq_not_cur(seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)
    #current frame
    cur_im = transform_rgb(cur_filename)
    #get annotation
    #anno_im = tf.cast(transform_anno(anno_filename), tf.int32)
    anno_im = transform_anno_test(anno_filename)

    return fuse_im, cur_im, ellip_info, anno_im, filename



#parse video data without annotations
def _parse_video_record(seq_filename_set, cur_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)
    #current frame
    cur_im = transform_rgb(cur_filename)
    
    return fuse_im, cur_im, filename
#parse video data without annotations using mask
def _parse_video_mask_record(seq_filename_set, cur_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq_not_cur(seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)
    #current frame
    cur_im =  tf.cast(transform_rgb(cur_filename), tf.float32)

    return fuse_im, cur_im, filename

#parse data for seq_mask+cur_frame(sequence annotaions as a mask)
def _parse_fusion_record(seq_filename_set, cur_filename, ellip_info, anno_filename, anno_soft_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq_mask(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)

    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)
    #get annotation
    anno_im = transform_anno_test(anno_filename)
    anno_soft_im = transform_anno_soft(anno_soft_filename)

    return fuse_im, current_frame, ellip_info, anno_im, anno_soft_im, filename


#-----------------get data------------------------
#-------------------------------------------------------------------
#1. get data for fusion: seq_mask(10) + cur_frame + annos_soft
def get_data_fusion(files, batch_size, if_cache, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        
        #1. read data list from records
        data = _read_images_list(files)

        #2. read image and batch dataset
        n_cpu_cores = os.cpu_count()

        data_list = data.map(_parse_fusion_record, num_parallel_calls=n_cpu_cores)
        data_batch = data_list.batch(batch_size)
        if if_cache:
            data_batch = data_batch.cache()
        data_batch = data_batch.repeat()
        #3. prefetch
        data_batch = data_batch.prefetch(5)
        
        #4. make iterator
        data_iter = data_batch.make_one_shot_iterator()
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, anno_soft_im_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+cfgs.cur_channel])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        ellip_info_batch.set_shape([batch_size, 4])
        anno_im_batch.set_shape([batch_size, im_size, im_size, cfgs.anno_channel])
        anno_soft_im_batch.set_shape([batch_size, im_size, im_size, cfgs.anno_soft_channel])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, anno_soft_im_batch, filename_batch


