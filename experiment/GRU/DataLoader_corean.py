#DataLoader for corean dataset.
import argparse, glob, os, cv2, sys, pickle, random, pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs


class DataLoader_c():
    def __init__(self, im_size, nbr_frames, anno_path, im_path):
        '''
        Args:
            nbf_frames: number of back off
        '''
        self.im_size = im_size
        self.dataset_size = cfgs.IMAGE_SIZE
        self.nbr_frames = nbr_frames

        self.anno_path = anno_path
        self.image_path = im_path

        #Get the last frame of video sequence which number is nbr_frames. 
        self.L = glob.glob(os.path.join(self.anno_path, '*.bmp'))

        random.shuffle(self.L)
        self.idx = 0

    def get_next_sequence(self):

        H, W = self.dataset_size
        h, w = self.im_size
        #offset = [np.random.randint(H - h),
            #np.random.randint(W - w)]
      
        offset = [0, 0]
        i0, j0 = offset
        i1, j1 = i0 + h, j0 + w

        im_path = self.L[self.idx % len(self.L)]
        self.idx += 1

        fn = os.path.splitext(im_path.split("/")[-1])[0]
        parts = fn.split('img')
        
        #File of which viede, no. of frames
        f, frame = parts[0], parts[1]

        images = []
        pad_images = []
        #Read gt
        gt = cv2.imread(im_path, 0)[i0:i1, j0:j1]/127
        #pdb.set_trace()
        gt = gt.astype(np.int64)
        
        #Read ims
        for dt in range(-self.nbr_frames + 1, 1):
            t = int(frame) + dt * cfgs.inter
            
            fn_ = '%simg%05d.bmp' % (f, t)
            frame_path = os.path.join(self.image_path, fn_)
            #np.newaxis: keep origin channel number
            im = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
            im = im.astype(np.float32)[i0:i1, j0:j1][np.newaxis, ...]
            
            im_pad = np.pad(im, ((0, 0), (2, 2), (0, 0), (0, 0)), 'constant')
   
            images.append(im)
            pad_images.append(im_pad)

        return images, pad_images, gt, fn
