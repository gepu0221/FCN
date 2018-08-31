#This code is for choose key frame using ellipse loss.
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
from six.moves import cPickle as pickle
import TensorflowUtils as utils
import read_data as scene_parsing
import datetime
import pdb
from BatchReader_multi_ellip import *
from BatchReader_multi_da import *
from key_frame_seq.generate_key_map import choose_key_frame
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

#from train_seq_parent import FCNNet
from train_seq_resnet_parent import Res101FCNNet as FCNNet

try:
    from .cfgs.config_key_map import cfgs
except Exception:
    from cfgs.config_key_map import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class SeqFCNNet(FCNNet):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir):

        FCNNet.__init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir)

        #seq
        self.seq_num = cfgs.seq_num
        self.channel = 3+self.seq_num
        self.inference_name = 'inference'
        self.images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, cfgs.seq_num+3], name='input_image')

        #key frame map
        self.key_map = {}
        
    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations, self.train_filenames = get_data_cache(self.train_records, self.batch_size, False, 'get_data_train')
            
            self.valid_images, self.valid_cur_ims, self.valid_ellip_infos, self.valid_annotations, self.valid_filenames = get_data_cache(self.valid_records, self.batch_size, False, 'get_data_valid_mask')

    def get_data_vis(self):
        with tf.device('/cpu:0'):
            self.vis_images, self.vis_cur_images, self.vis_ellip_infos, self.vis_annotations, self.vis_filenames, self.vis_init = get_data_vis(self.valid_records, self.batch_size)


    def get_data_video(self):
        with tf.device('/cpu:0'):
            self.video_images, self.video_cur_ims, self.video_filenames, self.video_init = get_data_video(self.valid_records, self.batch_size)

    #2. loss 
    def loss(self):
        self.logits = self.inference(self.images, self.inference_name, self.channel, self.keep_prob)
        self.pro = tf.nn.softmax(self.logits)
        self.pred_annotation = tf.expand_dims(tf.argmax(self.pro, dimension=3, name='pred'), dim=3)
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                        labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                        name='entropy_mask')))

        #test 
        #show the lower probility area
        self.pro_lower = tf.add(self.pro , cfgs.offset)
        self.pred_anno_lower = tf.expand_dims(tf.argmax(self.pro_lower, dimension=3, name='pred_lower'), dim=3)

    #3. generate key frame map for a batch
    def generate_key_map(self, im, filenames, pred_anno, gt_ellip_info):
        with tf.name_scope('key_map_batch'):
            sz_ = pred_anno.shape
            for i in range(sz_[0]):
                if_key = choose_key_frame(im[i], filenames[i], pred_anno[i], gt_ellip_info[i])
                self.key_map[filenames[i].strip().decode('utf-8')] = if_key
    
    #5. build graph
    
    def build(self):
        self.get_data_cache()
        self.loss()
        self.train_optimizer()
        self.summary()

    def build_vis(self):
        self.get_data_vis()
        self.loss()

    def build_video(self):
        self.get_data_video()
        self.loss()
    

    def generate_train_one_epoch(self, sess):
        try:
            count = 0
            while count<self.per_e_train_batch:
                count += 1
                images_, cur_ims, ellip_infos, annos_, filenames = sess.run([self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations, self.train_filenames])

                pred_anno = sess.run(self.pred_annotation,
                                      feed_dict={self.images: images_})
                self.generate_key_map(cur_ims.copy(), filenames, pred_anno.astype(np.uint8), ellip_infos)
        except tf.errors.OutOfRangeError:
            print('Error!')
     
    def generate_valid_one_epoch(self, sess):
        try:
            count = 0
            while count<self.per_e_valid_batch:
                count += 1
                images_, cur_ims, ellip_infos, annos_, filenames = sess.run([self.valid_images, self.valid_cur_ims, self.valid_ellip_infos, self.valid_annotations, self.valid_filenames])

                pred_anno = sess.run(self.pred_annotation,
                                      feed_dict={self.images: images_})
                
                self.generate_key_map(cur_ims.copy(), filenames, pred_anno.astype(np.uint8), ellip_infos)

        except tf.errors.OutOfRangeError:
            print('Error!')
    
    def generate(self):
        if not os.path.exists(self.logs_dir):
            print("The logs path '%s' is not found" % self.logs_dir)
            print("Create now..")
            os.makedirs(self.logs_dir)
            print("%s is created successfully!" % self.logs_dir)
        if not os.path.exists(cfgs.error_path):
            print("The check error path '%s' is not found" % cfgs.error_path)
            print("Create now..")
            os.makedirs(cfgs.error_path)
            print("%s is created successfully!" % cfgs.error_path)
        if not os.path.exists(cfgs.correct_path):
            print("The check correct path '%s' is not found" % cfgs.correct_path)
            print("Create now..")
            os.makedirs(cfgs.correct_path)
            print("%s is created successfully!" % cfgs.correct_path)



        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.device('/gpu:0'):
            with tf.Session(config=config) as sess:
                #1. initialize all variables
                sess.run(tf.global_variables_initializer())

                #2. Try to recover model
                saver = self.recover_model(sess)

                #3. Generate result data.
                self.generate_train_one_epoch(sess)
                self.generate_valid_one_epoch(sess)

    def pickle_key_map(self):
        pickle_path = os.path.join(cfgs.pickle_dir, cfgs.pickle_filename)
        if not os.path.exists(cfgs.pickle_dir):
            print('The pickle path %s is not found' % cfgs.pickle_dir)
            print('Create now..')
            os.makedirs(cfgs.pickle_dir)

        print('Pickling ...')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.key_map, f, pickle.HIGHEST_PROTOCOL)



#------------------------------------------------------------------------------

#Main function
def main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build()
        model.generate()
        model.pickle_key_map()



if __name__ == '__main__':
    main()
