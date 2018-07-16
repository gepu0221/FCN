#Batchreader for sequence FCN first modifed on 20180712.
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
    images = []   #combine with seq[7]
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

    def _read_images(self):
        
        self.images = np.array([self.fuse_seq(filename['current'], filename['seq']) for filename in self.files])
        self.count = 0
        if self.image_options.get("annotation", False) and self.image_options['annotation']:
            self.annotations = np.array(
            [np.expand_dims(self.transform_gray(filename['annotation']), axis=3) for filename in self.files])
            print('shape of annotations: ', self.annotations.shape)

        print('shape of images: ', self.images.shape)
        self.filenames = np.array(self.files)
        if self.image_options.get("visualize", False) and self.image_options["visualize"]:
            self._read_cur()

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
    

    #-----------------record read---------------------------
    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        if_finish = False
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            if_finish = True
            self.epochs_completed += 1
            #print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end], if_finish, self.epochs_completed

    def next_batch_with_name(self,batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            #Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images=self.images[perm]
            self.annotations = self.annotations[perm]
            self.filenames = self.filenames[perm]
            #Start next epoch
            start = 0
            self.batch_offset = batch_size
            
        end = self.batch_offset
        return self.images[start:end],self.annotations[start:end],self.filenames[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes],self.filenames[indexes]

    def get_random_batch_for_train(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def next_batch_valid(self,batch_size):
        #print('the number of images',self.images.shape[0])
        start=self.valid_batch_offset
        self.valid_batch_offset +=batch_size
        if_continue=True
        end=self.valid_batch_offset
        
        if self.valid_batch_offset> self.images.shape[0]:
            if_continue=False
            end=self.images.shape[0]
         
        return self.images[start:end],self.annotations[start:end],self.filenames[start:end], if_continue, start, end
    
    #Get batch including current frame for visualization when the input is combined sequence.
    def next_batch_val_vis(self,batch_size):

        start=self.valid_batch_offset
        self.valid_batch_offset +=batch_size
        if_continue=True
        end=self.valid_batch_offset
        
        if self.valid_batch_offset> self.images.shape[0]:
            if_continue=False
            end=self.images.shape[0]
         
        return self.images[start:end],self.annotations[start:end], self.filenames[start:end], self.cur_images[start:end], if_continue, start, end
     
    def next_batch_video_valid(self,batch_size):
        start=self.valid_batch_offset
        self.valid_batch_offset +=batch_size
        if_continue=True
        end=self.valid_batch_offset
        
        if self.valid_batch_offset> self.images.shape[0]:
            if_continue=False
            end=self.images.shape[0]
        return self.images[start:end], self.filenames[start:end], self.cur_images[start:end], if_continue, start, end
     
