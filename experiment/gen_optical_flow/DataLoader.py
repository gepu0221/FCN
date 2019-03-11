#Generate optical flow pickle data for gan training on 2019/03/11.
import argparse, glob, os, cv2, sys, pickle, random, pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs


class DataLoader():
    def __init__(self, im_size, nbr_frames, mask_path, im_path, da_im_path):
        '''
        Args:
            nbf_frames: number of back off
        '''
        self.dataset_size = cfgs.IMAGE_SIZE
        self.nbr_frames = cfgs.nbr_frames

        self.mask_path = mask_path
        self.image_path = im_path
        self.da_im_path = da_im_path

        #Get the last frame of video sequence which number is nbr_frames. 
        self.L = glob.glob(os.path.join(self.mask_path, '*.bmp'))
        random.shuffle(self.L)
        self.idx = 0
        
        
    def get_next_sequence(self):

        h, w = cfgs.inpt_resize_im_sz


      
        offset = [0, 0]
        i0, j0 = offset
        i1, j1 = i0 + h, j0 + w

        im_path = self.L[self.idx % len(self.L)]
        self.idx += 1

        fn = os.path.splitext(im_path.split("/")[-1])[0]
        fn_ori = fn.split('_')[0]
        parts = fn_ori.split('img')
        #File of which viede, no. of frames
        f, frame = parts[0], parts[1]

        fn_split = fn.split('_')
        fn = '%s_%s' % (fn_ori, fn_split[1])
        
        images = []
        flag = True
        #Read mask
        mask = cv2.imread(im_path)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = mask.astype(np.int64)
        mask_sum = np.sum(np.where(mask>127.5, 1, 0))
        mask = np.expand_dims(mask, 0)

        
        #Read ims pre
        dt = -1
        while True:
            t = int(frame) + dt * cfgs.inter
            
            if t < 0:
                flag = False
                break
            
            fn_ = '%simg%05d.bmp' % (f, t)
            frame_path = os.path.join(self.image_path, fn_)
            if os.path.exists(frame_path):
                #np.newaxis: keep origin channel number
                im = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                #pad for static network(u-net)
                im = np.expand_dims(im, axis=0)
                images.append(im)
                break
            else:
                dt -= 1
                continue
        
        #Read current
        t = int(frame)
        fn_cur = '%simg%05d.bmp' % (f, t)
        cur_frame_path = os.path.join(self.image_path, fn_cur)
        cur_im = cv2.imread(cur_frame_path, cv2.IMREAD_COLOR)
        cur_im = cv2.cvtColor(cur_im, cv2.COLOR_BGR2RGB)
        cur_im = cv2.resize(cur_im, (w, h), interpolation=cv2.INTER_CUBIC)
        #Resize for inpainting 256*256
        cur_im = np.expand_dims(cur_im, axis=0)
        images.append(cur_im)

        fn = '%simg%05d' % (f, t)

        return images, mask, fn, flag
