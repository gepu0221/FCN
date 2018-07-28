#Multithreading batchreader for sequence FCN first modifed on 20180712.
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
    current_frame = transform_rgb(cur_filename)
    frame = current_frame
    for i in range(0,cfgs.seq_num):
        frame = tf.concat((frame, transform_gray(seq_set_filename[i])), 2)
   
    return frame

def fuse_seq_not_cur(seq_set_filename):
 
    frame = transform_gray(seq_set_filename[0])
    for i in range(1, cfgs.seq_num):
        frame = tf.concat((frame, transform_gray(seq_set_filename[i])), 2)

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

#----------Read list Begin------------
def _read_images_list(files, shuffle=True):
    
    if shuffle:
        random.shuffle(files)

    #seq, cur, anno, filename
    seq_images = [item['seq'] for item in files]
    cur_images = [item['current'] for item in files]
    
    annotations = [item['annotation'] for item in files]
   
    filenames =  [item['filename'] for item in files]
    data = tf.data.Dataset.from_tensor_slices((seq_images, cur_images, annotations, filenames))

    return data

def _read_video_list(files, shuffle=True):
    
    if shuffle:
        ramdom.shuffle(files)

    #seq, cur, filename
    seq_images = [item['seq'] for item in files]
    cur_images = [item['current'] for item in files]
    filenames = [item['filename'] for item in files]

    data = tf.data.Dataset.from_tensor_slices((seq_images, cur_images, filenames))

    return data
#----------------Read List End----------------

#----------------Parse List Begin-------------
def _parse_record(seq_filename_set, cur_filename, anno_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)
    #get annotation
    anno_im = transform_anno_test(anno_filename)
    return fuse_im, anno_im, filename

def _parse_record_mask(seq_filename_set, cur_filename, anno_filename, filename):
    #fuse_im_seq(anot current)
    fuse_im = fuse_seq_not_cur(seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)

    current_frame = tf.cast(transform_rgb(cur_filename), tf.float32)

    anno_im = transform_anno_test(anno_filename)

    return fuse_im, current_frame, anno_im, filename



def _parse_vis_record(seq_filename_set, cur_filename, anno_filename, filename):
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fuse_im, tf.float32)
    #current frame
    cur_im = transform_rgb(cur_filename)
    #get annotation
    #anno_im = tf.cast(transform_anno(anno_filename), tf.int32)
    anno_im = transform_anno_test(anno_filename)

    return fuse_im, cur_im, anno_im, filename

#parse video data without annotations
def _parse_video_record(seq_filename_set, cur_filename, filename):
    
    #fuse sequence
    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    fuse_im = tf.cast(fues_im, tf.float32)
    #current frame
    cur_im = transform_rgb(cur_filename)
    
    return fuse_im, cur_im, filename
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
        fuse_im_batch, anno_im_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, anno_im_batch, filename_batch

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
        fuse_im_batch, cur_im_batch, anno_im_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num])
        cur_im_batch.set_shape([batch_size, im_size, im_size, cfgs.seq_num])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])
        
        return fuse_im_batch, cur_im_batch, anno_im_batch, filename_batch
#3. old get data for train using iterator
def get_data_(train_files, valid_files, batch_size, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:

        #1. read data list from records
        train_data = _read_images_list(train_files)
        #if cache
        #train_data = train_data.cache()
        valid_data = _read_images_list(valid_files)

        #2. read image through map(), and batch the dataset
        n_cpu_cores = os.cpu_count()
        
        #train
        train_list = train_data.map(_parse_record, num_parallel_calls=n_cpu_cores) # each item of train data is the param of fun _parse_record
        #*
        #train_list = train_list.cache()
        #train_list = train_list.repeat(2)
        #*
        train_batch = train_list.batch(batch_size)
        #train_batch = train_list.shuffle(buffer_size=13000).batch(batch_size)
        train_batch = train_batch.cache()
        train_batch = train_batch.repeat(2)
        #valid
        valid_list = valid_data.map(_parse_record, num_parallel_calls=n_cpu_cores)
        valid_batch = valid_list.batch(batch_size)
        #valid_batch = valid_batch.cache()
        
        #3. prefetch
        train_batch = train_batch.prefetch(5)
        valid_batch = valid_batch.prefetch(5)

        #4. create iterator
        iterator = tf.data.Iterator.from_structure(train_batch.output_types, train_batch.output_shapes) 
        #4.1 initialize different iterator
        train_init = iterator.make_initializer(train_batch, name='train_init')
        valid_init = iterator.make_initializer(valid_batch, name='valid_init')
        
        #5. fetch and return
        fuse_im_batch, anno_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, anno_im_batch, filename_batch, train_init, valid_init

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
        fuse_im_batch, cur_im_batch, anno_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return fuse_im_batch, cur_im_batch, anno_im_batch, filename_batch, vis_init



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
 

if __name__=='__main__':
    filename = tf.placeholder(dtype=tf.string, name='filename') 
    im_an = transform_anno_test(filename)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        im_ = sess.run([im_an], feed_dict={filename:'s2img00016.bmp'})
        print(im_[0].shape)
        cv2.imwrite('get_an_noint.jpg', im_[0])
        
        for i in range(224):
            for j in range(224):
                if im_[0][i][j]!=0:
                    print(im_[0][i][j])
    '''
    im_m = transform_misc('s2img00308.jpg')
    print(im_m)
    cv2.imwrite('get_m_an.jpg', im_m)
    '''


