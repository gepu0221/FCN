#Mulitthreading batchreader for sequence FCN first modifed on 20180712.
import numpy as np
import scipy.misc as misc
import cv2
import time


try:
    from .cfgs.config_data import cfgs 
except Exception:
    from cfgs.config_data import cfgs



class BatchDatset:
    files = []
    seq_images = []   #sequences[0..4] in combine with seq[7]
    cur_images = [] #current image only

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
        self._read_images()

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
        seq_frame = np.array([np.expand_dims(self.transform_gray(seq_filename), axis=2) for seq_filename in seq_set_filename])
        #print(current_frame.shape)
        #print(seq_frame.shape)
        if seq_frame.shape[0] == cfgs.seq_num:
            frame = current_frame
            for i in range(cfgs.seq_num):
                frame = np.concatenate((frame, seq_frame[i]), axis=2)
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
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)


    def transform_rgb(self, filename):
        image = cv2.imread(filename)
        if len(image.shape) < 3:
            image = np.array([image for i in range(3)])
            
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        else:
            resize_image = image

        return np.array(resize_image)
    
    def transform_gray(self, filename):
        image = cv2.imread(filename)
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception('Picture read failed!!!')
            
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = cv2.resize(gray_image, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        else:
            resize_image = gray_image

        return np.array(resize_image)

    def _read_images_list(self):
        
        self.seq_images = [item['seq'] for item in self.files]
        self.cur_images = [item['current'] for item in self.files]
        if self.image_option.get('annotation', False) and self.image_options['annotation']:
            self.annotations = [item['annotation'] for item in self.files]
        
        self.flienames = [item['filename'] for item in self.files]

        data = tf.data.Dataset.from_tensor_slices((self.seq_images, self.cur_images, self.annotations, self.filenames))

        return data

    def _parse_record(self, cur_filename, seq_filename_set, anno_filename=None, filename):
        
        #fuse sequence
        fuse_im = self.fuse_seq(cur_filename, seq_filename_set)
        
        #get annotation
        if self.image_options.get('annotation', False) and self.image_options['annotation']:
            anno_im = self.transform_gray(anno_filename)
        else:
            anno_im = None
        
        #get current frame
        if self.image_options.get('visualize', False) and self.image_options['visualize']:
            cur_im = self._read_cur(cur_filename)
        else:
            cur_im = None

        return fuse_im, anno_im, cur_im, filename

        

    def get_data_from_filelist(self, batch_size, variable_scope_name='get_data'):
        
        with tf.variable_scope(variable_scope_name) as var_scope:
            data = self._read_images_list()

            n_cpu_cores = os.cpu_count()

            data_list = data.map(lambda fuse_im, anno_im, cur_im, filename: _parse_record(fuse_im, anno_im, cur_im, filename), num_parallel_calls=n_cpu_cores)
            data_batch = data_list.batch(batch_size)

            data_batch = data_batch.prefetch(5)

            iterator = tf.data.Iterator.from_structure(data_batch.output_types, data_batch.output_shapes) 
            data_init = iterator.make_initializer(data_batch, name='data_init')

            fuse_im_batch, anno_im_batch, cur_im_batch, filename_batch = iterator.get_next()
            im_size = int(self.image_options['resize_size'])
            fuse_im_batch.set([ batch_size, im_size, im_size, cfgs.seq_num+3])
        
        return fuse_im_batch, anno_im_batch, cur_im_batch, filename_batch, data_init
