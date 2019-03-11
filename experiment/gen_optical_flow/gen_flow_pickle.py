from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import models.TensorflowUtils as utils
import models.G_Layers as utils_layers
import datetime
import pdb
#import tool.CaculateAccurary as accu
from six.moves import xrange
import shutil
#Pretrain model
from tensorflow.python.framework import ops
from DataLoader import DataLoader
from models.flownet2 import Flownet2


try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs


class FlowGen(object):
    
    def __init__(self):
        
        self.Pre_Net()

    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            #train data loader
            self.train_dl = DataLoader(cfgs.IMAGE_SIZE, cfgs.nbr_frames, cfgs.train_mask_path, cfgs.image_path, cfgs.da_im_path)
            #valid data loader
            self.valid_dl = DataLoader(cfgs.IMAGE_SIZE, cfgs.nbr_frames, cfgs.val_mask_path, cfgs.image_path, cfgs.da_im_path)


    #2. Net
    def Pre_Net(self):
        
        '''
        Prepare all net models.
        '''

        #Input 
        sz = [1, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 3]
        re_sz = [cfgs.RE_IMAGE_SIZE[0], cfgs.RE_IMAGE_SIZE[1]]
        self.prev_img = tf.placeholder(tf.float32, shape=sz)
        self.cur_img = tf.placeholder(tf.float32, shape=sz)

        self.re_prev_img = tf.image.resize_bicubic(self.prev_img, re_sz)
        self.re_cur_img = tf.image.resize_bicubic(self.cur_img, re_sz)


        #1. Load bilinear warping model.
        self.bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
        @ops.RegisterGradient("BilinearWarping")
        def _BilinearWarping(op, grad):
            return self.bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])

        with tf.variable_scope('flow'):
            self.flow_network = Flownet2(self.bilinear_warping_module)
            self.flow_tensor = self.flow_network(self.re_cur_img, self.re_prev_img, flip=True)
    
    #3. build 
    def build(self):
        
        self.get_data_cache()
        self.create_re_dir()
        
    #4. Model recover and save    
    def return_saver(self, sess, logs_dir, model_name, var_list):
        saver = tf.train.Saver(var_list)
        if os.path.exists(logs_dir):
            saver.restore(sess, os.path.join(logs_dir, model_name))
            print('Model %s restore finished' % logs_dir)

        return saver


    def recover_model(self, sess):
        var_list = tf.trainable_variables()
        var_flow = [k for k in var_list if k.name.startswith('flow')]

        loader_flow = self.return_saver(sess, cfgs.flow_logs_dir, cfgs.flow_logs_name, var_flow)

    #5. Create save path.
    def create_re_dir(self):
        if not os.path.exists(cfgs.save_path):
            os.makedirs(cfgs.save_path)
        self.train_save_dir = os.path.join(cfgs.save_path, 'train')
        self.valid_save_dir = os.path.join(cfgs.save_path, 'valid')
        if not os.path.exists(self.train_save_dir):
            os.makedirs(self.train_save_dir)
        if not os.path.exists(self.valid_save_dir):
            os.makedirs(self.valid_save_dir)
        self.train_flist = cfgs.train_file_fn
        self.valid_flist = cfgs.valid_file_fn
        if not os.path.exists(cfgs.flist_save_path):
            os.makedirs(cfgs.flist_save_path)
        if not os.path.exists(self.train_flist):
            os.mknod(self.train_flist)
        if not os.path.exists(self.valid_flist):
            os.mknod(self.valid_flist)

    
    def gen_train(self, sess, data_loader, data_num):
        '''
        Generate optical flow once of all train data.
        '''
        #fo = open(self.train_flist, 'w')
        file_names = []

        for count in range(1, data_num):

            images, mask, fn, flag = data_loader.get_next_sequence()
            if flag == False:
                print(fn)
                # Can't find prev data.
                continue

            im, last_im = images[1], images[0]
            flow = sess.run(self.flow_tensor,
                            feed_dict={self.cur_img: im,
                                      self.prev_img: last_im})
            
            flow_fn = os.path.join(self.train_save_dir, fn+'.npy')
            file_names.append(flow_fn)
            np.save(flow_fn, flow)
        
        fo = open(file_names, 'w')
        fo.write('\n'.join(flow))
        fo.close()
        



    def gen(self):
    
        print('The graph path is %s' % cfgs.flow_logs_dir)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.device('/gpu:0'):
            with tf.Session(config=config) as sess:
                #1. initialize all variables
                sess.run(tf.global_variables_initializer())

                #2. Try to recover model
                self.recover_model(sess)

                self.gen_train(sess, self.train_dl, cfgs.train_num)

                #self.gen_valid(sess, self.valid_dl, cfgs.valid_num)


def main():
 
    with tf.device('/gpu:0'):
        #train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        #print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = FlowGen()
        model.build()
        model.gen()



if __name__ == '__main__':
    main()
