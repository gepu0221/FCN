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

    #cur, anno, filename
    cur_images = [item['current'] for item in files]
    
    annotations = [item['annotation'] for item in files]
   
    filenames =  [item['filename'] for item in files]
    data = tf.data.Dataset.from_tensor_slices((cur_images, annotations, filenames))

    return data

def _read_video_list(files, shuffle=True):
    
    if shuffle:
        random.shuffle(files)

    #cur, filename
    cur_images = [item['current'] for item in files]
    filenames = [item['filename'] for item in files]

    data = tf.data.Dataset.from_tensor_slices((cur_images, filenames))

    return data
#----------------Read List End----------------

#----------------Parse List Begin-------------
def _parse_record(cur_filename, anno_filename, filename):
    
    #fuse sequence
    im = tf.cast(transform_rgb(cur_filename), tf.float32)
    #get annotation
    anno_im = transform_anno_test(anno_filename)
    return im, anno_im, filename


def _parse_vis_record(cur_filename, anno_filename, filename):
  
    #current frame
    im = transform_rgb(cur_filename)
    #get annotation
    anno_im = transform_anno_test(anno_filename)

    return im, anno_im, filename


#parse video data without annotations
def _parse_video_record(cur_filename, filename):
    
    #current frame
    im = transform_rgb(cur_filename)
    
    return im, filename

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
        im_batch, anno_im_batch, filename_batch = data_iter.get_next()

        im_size = int(image_options['resize_size'])
        im_batch.set_shape([ batch_size, im_size, im_size, 3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return im_batch, anno_im_batch, filename_batch

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
        im_batch, anno_im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        cur_im_batch.set_shape([batch_size, im_size, im_size, 3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])

    return im_batch, anno_im_batch, filename_batch, vis_init


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
        im_batch, filename_batch = iterator.get_next()

        im_size = int(image_options['resize_size'])
        im_batch.set_shape([batch_size, im_size, im_size,3])
        filename_batch.set_shape([batch_size])

    return im_batch, filename_batch, video_init

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


