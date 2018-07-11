import numpy as np
import scipy.misc as misc
import cv2

try:
    from .cfgs.config_data import cfgs 
except Exception:
    from cfgs.config_data import cfgs



class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}

    def __init__(self, records_list, image_options={}):
        print("Initializing Batch Dataset Reader...")
        print('image_options: ', image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        
        self.images = np.array([self.fuse_seq(filename['current'], filename['seq']) for filename in self.files])
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print('shape of images: ', self.images.shape)
        print('shape of annotations: ', self.annotations)
        self.filenames = np.array(self.files)

    #fuse current frame with sequence 
    def fuse_seq(self, cur_filename, seq_set_filename):
        current_frame = self.transform_rgb(cur_filename)
        seq_frame = np.array([np.expand_dims(self.transform_gray(seq_filename), axis=2) for seq_filename in seq_set_filename])
        #print(current_frame.shape)
        #print(seq_frame.shape)
        if seq_frame.shape[0] == cfgs.seq_num:
            frame = current_frame
            for i in range(cfgs.seq_num):
                frame = np.concatenate((frame, seq_frame[i]), axis=2)
            #print(frame.shape)
        return frame

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

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]
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
            
