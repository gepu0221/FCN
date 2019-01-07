#DataLoader for corean dataset to read data using for sequence data test on 2019/01/06.
import argparse, glob, os, cv2, sys, pickle, random, pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs


class DataLoader_c():
    def __init__(self):
        '''
        Args:
        '''
        #Get the images need reconstruction
        self.L = glob.glob(os.path.join(cfgs.reconst_im_path, '*.bmp'))

        #random.shuffle(self.L)
        self.idx = 0

        self.ratio = [cfgs.IMAGE_SIZE[0]/540, cfgs.IMAGE_SIZE[1]/960]

    def get_ellip_info(self, f_s):
        '''
            Args:
                f_s: the split of filename
        '''
        cx = int(f_s[1]) * self.ratio[1]
        cy = int(f_s[2]) * self.ratio[0]
        w = int(f_s[3])
        h = int(f_s[4])

        ellip_info = [cx, cy, w*self.ratio[0], h*self.ratio[1]]

        return ellip_info

    def get_next_sequence(self):


      
        im_path = self.L[self.idx % len(self.L)]
        self.idx += 1

        fn = os.path.splitext(im_path.split("/")[-1])[0]
        fn_ori = fn.split('_')[0]
        parts = fn_ori.split('img')
        #File of which viede, no. of frames
        f, frame = parts[0], parts[1]

        fn_split = fn.split('_')
        fn = fn_ori
        
        cur_im = cv2.imread(im_path)
        cur_im = cv2.cvtColor(cur_im, cv2.COLOR_BGR2RGB)


        prev_im_path = os.path.join(cfgs.key_im_path, fn_ori+'.bmp')
        prev_im = cv2.imread(prev_im_path)
        prev_im = cv2.cvtColor(prev_im, cv2.COLOR_BGR2RGB)


        ellip_info = self.get_ellip_info(fn_split)

        cur_im = np.expand_dims(cur_im, axis=0)
        prev_im = np.expand_dims(prev_im, axis=0)

        return cur_im, prev_im, ellip_info, fn
        
