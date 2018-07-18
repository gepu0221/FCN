#Multithreading batchreader for sequence FCN first modifed on 20180712.
import numpy as np
import tensorflow as tf
import scipy.misc as misc
import os
import time


try:
    from .cfgs.config_data import cfgs 
except Exception:
    from cfgs.config_data import cfgs



class BatchDatset:
    files = []
    seq_images = []   #sequences[0..4] in combine with seq[7]
    cur_images = [] #current image only
    filenames = []

    annotations = []
    image_options = {}
    batch_offset = 0
    valid_batch_offset = 0
    epochs_completed = 0

    #
    count =0

    def __init__(self, records_list, image_options={}):
        print("Initializing Batch Dataset Reader...")
        print('image_options: ', image_options)
        self.files = records_list
        self.image_options = image_options
        #self._read_images()

    def _read_cur(self):
        self.cur_images = np.array([self.transform_current(filename['current']) for filename in self.files])

    #read current frame only for validation visualize
    def transform_current(self, cur_filename):
        current_frame = self.transform_misc(cur_filename)

        return current_frame


    #fuse current frame with sequence 
    def fuse_seq(self, cur_filename, seq_set_filename):
        #
        self.count += 1
        current_frame = self.transform_rgb(cur_filename)
        frame = current_frame
        for i in range(0,cfgs.seq_num):
            frame = tf.concat((frame, self.transform_gray(seq_set_filename[i])), 2)
        if self.count % 1000 == 0:
            print('--image%d--' % (self.count))
        return frame

    def transform_misc(self, filename):
        #print('image_name',filename)
        image = misc.imread(filename)
        if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,[resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)


    def transform_rgb(self, filename):
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, channels=3)
            
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = tf.image.resize_bilinear([image], size=[resize_size, resize_size])[0]
        else:
            resize_image = image

        return resize_image
    
    def transform_gray(self, filename):
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, channels=1)
            
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = tf.image.resize_bilinear([image], size=[resize_size, resize_size])[0]
        else:
            resize_image = gray_image

        return resize_image

    def _read_images_list(self):
        
        #seq, cur, anno, filename
        self.seq_images = [item['seq'] for item in self.files]
        self.cur_images = [item['current'] for item in self.files]
        if self.image_options.get('annotation', False) and self.image_options['annotation']:
            print('anno exists')
            self.annotations = [item['annotation'] for item in self.files]
       
        self.filenames =  [item['filename'] for item in self.files]


        data = tf.data.Dataset.from_tensor_slices((self.seq_images, self.cur_images, self.annotations, self.filenames))

        return data

    def _parse_record(self, seq_filename_set, cur_filename, anno_filename, filename):
        
        #fuse sequence
        fuse_im = self.fuse_seq(cur_filename, seq_filename_set)

        print('fuse_im', tf.shape(fuse_im))
        #get annotation
        anno_im = tf.cast(self.transform_gray(anno_filename), tf.int32)
        print('anno_im', tf.shape(anno_im))

        return fuse_im, anno_im, filename

        

    def get_data_from_filelist(self, batch_size, variable_scope_name='get_data'):
        
        with tf.variable_scope(variable_scope_name) as var_scope:
            data = self._read_images_list()

            n_cpu_cores = os.cpu_count()

            data_list = data.map(self._parse_record, num_parallel_calls=n_cpu_cores)
            data_batch = data_list.batch(batch_size)

            data_batch = data_batch.repeat()

            data_batch = data_batch.prefetch(5)

            iterator = tf.data.Iterator.from_structure(data_batch.output_types, data_batch.output_shapes) 
            data_init = iterator.make_initializer(data_batch, name='data_init')

            fuse_im_batch, anno_im_batch, filename_batch = iterator.get_next()
            print('shape of fuse_batch', tf.shape(fuse_im_batch)[1])
            im_size = int(self.image_options['resize_size'])
            fuse_im_batch.set_shape([ batch_size, im_size, im_size, cfgs.seq_num+3])
        
        return fuse_im_batch, anno_im_batch, filename_batch, data_init
