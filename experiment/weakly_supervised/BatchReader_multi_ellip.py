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
    from .cfgs.config_train_u_net import cfgs 
except Exception:
    from cfgs.config_train_u_net import cfgs
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

def transform_rgb(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
        
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = cfgs.RESIZE_IMAGE_SIZE
        resize_image = tf.image.resize_bilinear([image], size=[resize_size[0], resize_size[1]])[0]
    else:
        resize_image = image

    paddings = tf.constant([[2,2], [0,0], [0,0]])
    resize_image = tf.pad(resize_image, paddings, 'CONSTANT')

    return resize_image

def transform_gray(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
        
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = cfgs.IMAGE_SIZE
        resize_image = tf.image.resize_bilinear([image], size=[resize_size[0], resize_size[1]])[0]
    else:
        resize_image = image

    return resize_image

def transform_anno(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image)
    
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = cfgs.ANNO_IMAGE_SIZE
        #resize_image = tf.image.resize_bilinear([image], size=[resize_size[0], resize_size[1]])[0]
        #Nearest_neighbor
        resize_image = tf.image.resize_nearest_neighbor([image], size=[resize_size[0], resize_size[1]])[0]
    else:
        resize_image = image
    
    sz = [cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1]]
    im_comp = tf.ones(sz, dtype=tf.int32)
    resize_image = resize_image[:, :, 0]
    resize_image = tf.where(tf.less_equal(resize_image, 0), 1-im_comp, im_comp)
    resize_image = tf.expand_dims(resize_image, 2)
  
    
    return resize_image

def transform_anno_test(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image)
    image = tf.cast(tf.expand_dims(tf.reduce_mean(image, 2),2), tf.int32)
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
    label = [item['label'] for item in files]

    filenames = [item['filename'] for item in files]
    data = tf.data.Dataset.from_tensor_slices((seq_images, cur_images, ellip_infos, annotations, label, filenames))

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
def _parse_record(seq_filename_set, cur_filename, ellip_info, anno_filename, label, filename):
    
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)

    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)
    #get annotation
    anno_im = transform_anno(anno_filename)
    
    return fuse_im, current_frame, ellip_info, anno_im, label, filename

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
def _parse_seq_mask_record(seq_filename_set, cur_filename, ellip_info, anno_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq_mask(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)

    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)
    #get annotation
    anno_im = transform_anno_test(anno_filename)
    return fuse_im, current_frame, ellip_info, anno_im, filename


#-----------------get data------------------------
#1. get data for train
def get_data_cache(files, batch_size, if_cache, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        
        #1. read data list from records
        data = _read_images_list(files)

        #2. read image and batch dataset
        n_cpu_cores = os.cpu_count()

        data_list = data.map(_parse_record, num_parallel_calls=n_cpu_cores)
        data_batch = data_list.batch(batch_size)
        if if_cache:
            data_batch = data_batch.cache()
        data_batch = data_batch.repeat()
        #3. prefetch
        data_batch = data_batch.prefetch(5)
        
        #4. make iterator
        data_iter = data_batch.make_one_shot_iterator()
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, label_batch, filename_batch = data_iter.get_next()

        im_size = cfgs.IMAGE_SIZE
        anno_size = cfgs.ANNO_IMAGE_SIZE
        fuse_im_batch.set_shape([ batch_size, im_size[0], im_size[1], cfgs.seq_num+cfgs.cur_channel])
        cur_im_batch.set_shape([batch_size, im_size[0], im_size[1], 3])
        ellip_info_batch.set_shape([batch_size, 4])
        anno_im_batch.set_shape([batch_size, anno_size[0], anno_size[1], 1])
        label_batch.set_shape([batch_size])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, label_batch, filename_batch

#2. get data for mask
def get_data_mask(files, batch_size, if_cache, variable_scope_name='get_data_mask'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        
        #1. read data list from records
        data = _read_images_list(files)

        #2. read image and batch dataset
        n_cpu_cores = os.cpu_count()

        data_list = data.map(_parse_record_mask, num_parallel_calls=n_cpu_cores)

        data_batch = data_list.batch(batch_size)
        if if_cache:
            data_batch = data_batch.cache()
        data_batch = data_batch.repeat()
        #3. prefetch
        data_batch = data_batch.prefetch(5)
        
        #4. make iterator
        data_iter = data_batch.make_one_shot_iterator()
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([batch_size, im_size, im_size, cfgs.seq_num])
        ellip_info_batch.set_shape([batch_size, 4])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])
        
        return fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch


def get_data_vis(vis_files, batch_size, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:

        #1. read data list from records
        vis_data = _read_images_list(vis_files)

        #2. read image through map(), and batch the dataset
        n_cpu_cores = os.cpu_count()
        
        #vis
        vis_list = vis_data.map(_parse_vis_record, num_parallel_calls=n_cpu_cores) # each item of train data is the param of fun _parse_record
        vis_batch = vis_list.batch(batch_size)
        
        #3. prefetch
        vis_batch = vis_batch.prefetch(5)

        #4. create iterator
        iterator = tf.data.Iterator.from_structure(vis_batch.output_types, vis_batch.output_shapes) 
        #4.1 initialize iterator
        vis_init = iterator.make_initializer(vis_batch, name='vis_init')
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+cfgs.cur_channel])
        ellip_info_batch.set_shape([batch_size, 4])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch, vis_init

def get_data_vis_mask(vis_files, batch_size, variable_scope_name='get_data_vis_mask'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        
        #1. read data list from records
        vis_data = _read_images_list(vis_files)

        #2. read image through map(), and batch the dataset
        n_cpu_cores = os.cpu_count()

        #vis 
        vis_list = vis_data.map(_parse_vis_mask_record, num_parallel_calls=n_cpu_cores)
        vis_batch = vis_list.batch(batch_size)

        #3. prefetch
        vis_batch = vis_batch.prefetch(5)

        #4. create iterator
        iterator = tf.data.Iterator.from_structure(vis_batch.output_types, vis_batch.output_shapes) 
        #4.1 initialize iterator
        vis_init = iterator.make_initializer(vis_batch, name='vis_init')
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        ellip_info_batch.set_shape([batch_size, 1])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch, vis_init


#get data from video which has no annotations
def get_data_video(video_files, batch_size, variable_scope_name='get_data_video'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:

        #1. read data list from vido records
        video_data = _read_video_list(video_files)

        #2. read image through map(), and batch the dataset
        n_cpu_cores = os.cpu_count()
        
        #video
        video_list = video_data.map(_parse_video_record, num_parallel_calls=n_cpu_cores) # each item of video data is the param of fun _parse_record
        video_batch = video_list.batch(batch_size)
       
        #3. prefetch
        video_batch = video_batch.prefetch(5)

        #4. create iterator
        iterator = tf.data.Iterator.from_structure(video_batch.output_types, video_batch.output_shapes) 
        #4.1 initialize different iterator
        video_init = iterator.make_initializer(video_batch, name='video_init')
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        cur_im_batch.set_shape([batch_size, im_size, im_size,3])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, filename_batch, video_init

def get_data_video_mask(video_files, batch_size, variable_scope_name='get_data_vide0_mask'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        
        #1. read data list from video records
        video_data = _read_video_list(video_files)

        #2. read image through map(), and batch the dataset
        n_cpu_cores = os.cpu_count()

        #video
        video_list = video_data.map(_parse_video_mask_record, num_parallel_calls=n_cpu_cores)
        video_batch = video_list.batch(batch_size)

        #3. prefetch
        video_batch = video_batch.prefetch(5)

        #4. create iterator
        iterator = tf.data.Iterator.from_structure(video_batch.output_types, video_batch.output_shapes) 
        #4.1 initialize different iterator
        video_init = iterator.make_initializer(video_batch, name='video_init')
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num])
        cur_im_batch.set_shape([batch_size, im_size, im_size,3])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, filename_batch, video_init


#-----------------------------------------------------------------
#3. get data for result recover(seq+cur)
def get_data_cache_re(files, batch_size, if_cache, variable_scope_name='get_data_recover'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:

        #1. read data list from records
        data = _read_images_list_re(files)

        #2. read images and batch dataset
        n_cpu_cores = os.cpu_count()

        data_list = data.map(_parse_record_re, num_parallel_calls=n_cpu_cores)
        data_batch = data_list.batch(batch_size)
        if if_cache:
            data_batch = data_batch.cache()
        data_batch = data_batch.repeat()
        #3. prefetch
        data_batch = data_batch.prefetch(5)

        #4. make iterator
        data_iter = data_batch.make_one_shot_iterator()

        #5. fetch and return
        cur_im_batch, ellip_info_batch, anno_im_batch, re_recover_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        #fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        ellip_info_batch.set_shape([batch_size, 4])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        re_recover_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return cur_im_batch, ellip_info_batch, anno_im_batch, re_recover_batch, filename_batch

#-------------------------------------------------------------------
#4. get data for seq_mask(10) + cur_frame
def get_data_seq_mask(files, batch_size, if_cache, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        
        #1. read data list from records
        data = _read_images_list(files)

        #2. read image and batch dataset
        n_cpu_cores = os.cpu_count()

        data_list = data.map(_parse_seq_mask_record, num_parallel_calls=n_cpu_cores)
        data_batch = data_list.batch(batch_size)
        if if_cache:
            data_batch = data_batch.cache()
        data_batch = data_batch.repeat()
        #3. prefetch
        data_batch = data_batch.prefetch(5)
        
        #4. make iterator
        data_iter = data_batch.make_one_shot_iterator()
        
        #5. fetch and return
        fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+cfgs.cur_channel])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        ellip_info_batch.set_shape([batch_size, 4])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, ellip_info_batch, anno_im_batch, filename_batch


