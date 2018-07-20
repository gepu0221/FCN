#Multithreading batchreader for sequence FCN first modifed on 20180712.
import numpy 
import tensorflow as tf
import scipy.misc as misc
import os
import random
import pdb
import time
import cv2
import read_data as scene_parsing

try:
    from .cfgs.config_data import cfgs 
except Exception:
    from cfgs.config_data import cfgs
image_options = {'resize':True, 'resize_size':cfgs.IMAGE_SIZE}

    #fuse current frame with sequence 
def fuse_seq(cur_filename, seq_set_filename):
    #
    current_frame = transform_rgb(cur_filename)
    frame = current_frame
    for i in range(0,cfgs.seq_num):
        frame = tf.concat((frame, transform_gray(seq_set_filename[i])), 2)
   
    return frame

def transform_misc(filename):
    #print('image_name',filename)
    image = misc.imread(filename)
    if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        image = np.array([image for i in range(3)])

    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = misc.imresize(image,[resize_size, resize_size], interp='nearest')
    else:
        resize_image = image

    return np.array(resize_image)


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
        resize_image = gray_image

    return resize_image

def transform_anno(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1)
        
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = tf.image.resize_bilinear([image], size=[resize_size, resize_size])[0]
    else:
        resize_image = gray_image

    return resize_image



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

def _parse_record(seq_filename_set, cur_filename, anno_filename, filename):
    
    #fuse sequence

    fuse_im = fuse_seq(cur_filename, seq_filename_set)
    #fuse_im = transform_rgb(cur_filename)
    fuse_im = tf.cast(fuse_im, tf.float32)
    print('fuse_im', tf.shape(fuse_im))
    #get annotation
    #anno_im = tf.cast(transform_gray(anno_filename), tf.int32)
    anno_im = tf.cast(transform_anno(anno_filename), tf.int32)
    print('anno_im', tf.shape(anno_im))
    return fuse_im, anno_im, filename

    

def get_data_from_filelist(files, batch_size, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        data = _read_images_list(files)

        n_cpu_cores = os.cpu_count()

        data_list = data.map(_parse_record, num_parallel_calls=n_cpu_cores)
        data_batch = data_list.batch(batch_size)

        data_batch = data_batch.prefetch(5)

        iterator = tf.data.Iterator.from_structure(data_batch.output_types, data_batch.output_shapes) 
        print(data_batch.output_types, data_batch.output_shapes)
        data_init = iterator.make_initializer(data_batch, name='data_init')
        
        fuse_im_batch, anno_im_batch, filename_batch = iterator.get_next()
        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])
    return fuse_im_batch, anno_im_batch, filename_batch, data_init

def get_data_(train_files, valid_files, batch_size, variable_scope_name='get_data'):
    
    with tf.variable_scope(variable_scope_name) as var_scope:
        train_data = _read_images_list(train_files)
        valid_data = _read_images_list(valid_files)

        n_cpu_cores = os.cpu_count()
        
        #train
        train_list = train_data.map(_parse_record, num_parallel_calls=n_cpu_cores)
        train_batch = train_list.batch(batch_size)
        #valid
        valid_list = valid_data.map(_parse_record, num_parallel_calls=n_cpu_cores)
        valid_batch = valid_list.batch(batch_size)

        train_batch = train_batch.prefetch(5)
        valid_batch = valid_batch.prefetch(5)

        iterator = tf.data.Iterator.from_structure(train_batch.output_types, train_batch.output_shapes) 
        
        train_init = iterator.make_initializer(train_batch, name='train_init')
        valid_init = iterator.make_initializer(valid_batch, name='valid_init')

        fuse_im_batch, anno_im_batch, filename_batch = iterator.get_next()
        im_size = int(image_options['resize_size'])
        fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        anno_im_batch.set_shape([batch_size, im_size, im_size, 1])
        filename_batch.set_shape([batch_size])
    return fuse_im_batch, anno_im_batch, filename_batch, train_init, valid_init
 
if __name__=='__main__':
    train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
    train_data = _read_images_list(train_records)

