#This is the base class FCN for sequence 
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import TensorflowUtils as utils
import G_Layers as utils_layers
#import read_data as scene_parsing
import datetime
import pdb
#import BatchReader_multi as dataset
from BatchReader_multi_ellip import get_data_cache
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

try:
    from .cfgs.config_train_u_net import cfgs
except Exception:
    from cfgs.config_train_u_net import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class U_Net(object):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.NUM_OF_CLASSESS = n_classes
        self.IMAGE_SIZE = im_sz
        self.graph = tf.get_default_graph()
        self.lr= tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.learning_rate = float(init_lr)
        self.mode = mode
        self.logs_dir = logs_dir
        self.current_itr_var = tf.Variable(0, dtype=tf.int32, name='current_itr', trainable=True)
        self.cur_epoch = tf.Variable(1, dtype=tf.int32, name='cur_epoch', trainable=False)
        self.cur_batch_size = tf.placeholder(dtype=tf.int32, name='cur_batch_size')


        self.train_records = train_records
        self.valid_records = valid_records
        self.per_e_train_batch = len(self.train_records)/self.batch_size
        self.per_e_valid_batch = len(self.valid_records)/self.batch_size
        
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.input_keep_prob = tf.placeholder(tf.float32, shape=[], name='input_keep_prob')
        self.images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], cfgs.seq_num+cfgs.cur_channel], name='input_image')
        self.class_labels = tf.placeholder(tf.int32, shape=[None], name='class_occlusion_label')
        self.annotations = tf.placeholder(tf.int64, shape=[None, cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1], 1], name='annotations')

        if self.mode == 'visualize' or 'vis_video':
            self.result_dir = cfgs.result_dir
        self.at = cfgs.at
        self.gamma = cfgs.gamma

        #init U-Net
        self.u_net = utils_layers.U_Net_gp()

        #Hausdorff distance
        self.pos_m = tf.placeholder(tf.int32, shape=[None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 1, 2], name='position_matrix')
        self.gt_key = tf.placeholder(tf.int32, shape=[None, 1, 1, None, 2], name='gt_key_point')
        self.max_dist = math.sqrt(self.IMAGE_SIZE[0]**2 + self.IMAGE_SIZE[1]**2)
        self.eps = cfgs.eps
        self.alpha = cfgs.alpha

        
    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_annotations, self.train_filenames = get_data_cache(self.train_records, self.batch_size, False, 'get_data_train')
            self.valid_images, self.valid_annotations, self.valid_filenames = get_data_cache(self.valid_records, self.batch_size, False, 'get_data_valid')

    def get_data(self):
        with tf.device('/cpu:0'):
            self.images, self.annotations, self.filenames, self.train_init, self.valid_init = get_data_(self.train_records, self.valid_records, self.batch_size)
    
    def get_data_vis(self):
        with tf.device('/cpu:0'):
            self.vis_images, self.vis_cur_ims, self.vis_annotations, self.vis_filenames, self.vis_init = get_data_vis(self.valid_records, self.batch_size)

    def get_data_video(self):
        with tf.device('/cpu:0'):
            self.video_images, self.video_cur_ims, self.video_filenames, self.video_init = get_data_video(self.valid_records, self.batch_size)

    #2. net 
    def inference(self, images, inference_name, channel, keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up resnet101 initialized conv layers ...")
        #mean_pixel = np.mean(mean, axis=(0, 1))

        processed_images = utils.process_image(images, cfgs.mean_pixel)

        processed_images = tf.nn.dropout(processed_images, self.input_keep_prob)

        with tf.variable_scope(inference_name):
            
            #U-Net
            logits, class_logits = self.u_net.u_net_op(x=processed_images, 
                                         keep_prob_=keep_prob, 
                                         channels=channel,
                                         n_class = cfgs.n_class,
                                         layers = cfgs.layers,
                                         class_convs_num = cfgs.class_convs_num,
                                         features_root=cfgs.features_root,
                                         filter_size = cfgs.filter_size,
                                         pool_size = cfgs.pool_size)

            annotation_pred = tf.argmax(logits, dimension=3, name="prediction")
            print('logits shape', logits.shape)

        return logits, class_logits
     

    def infer(self):
        self.logits, self.class_logits = self.inference(self.images, 'inference_name', 7, self.keep_prob )

        #3. optmizer
    def train_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        var_list = tf.trainable_variables()
        #import pprint
        #pprint.pprint(var_list)
        #pdb.set_trace()
        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        grads = optimizer.compute_gradients(self.loss, var_list=var_list)
        self.train_op = optimizer.apply_gradients(grads)

    
    #4. loss
    def loss(self):
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                          name="entropy")))
        self.pred_annotation = tf.expand_dims(tf.argmax(self.pro, dimension=3, name='pred'), dim=3)

        self.pro = tf.nn.softmax(self.logits)

        #focal loss
        a_w = (1 - 2*self.at) * tf.cast(tf.squeeze(self.annotations, squeeze_dims=[3]), tf.float32) + self.at
        self.pro = tf.nn.softmax(self.logits)
      
        loss_weight = tf.pow(1-tf.reduce_sum(self.pro * tf.one_hot(tf.squeeze(self.annotations, squeeze_dims=[3]), self.NUM_OF_CLASSESS), 3), self.gamma)
     
    
        self.focal_loss = tf.reduce_mean(loss_weight * a_w * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                       labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                       name="entropy"))

    
    #4.1 weighted-hausdorff-loss
    def h_loss(self):
        '''
        Args:Compute weighted Hausdorff Distance function
        between the estimated probalility map and ground truth points.
        '''
        repeat_num = tf.shape(self.gt_key)[0]
        normalized_x = tf.tile(self.pos_m, [1,1,1,repeat_num,1])
        normalized_y = tf.tile(self.gt_key, [1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1,1])
        
        differences = tf.subtract(normalized_x, normalized_y)
        d_matrix = tf.cast(tf.reduce_sum(tf.pow(differences, 2), 4), tf.float32)
        
        p_est_pts = tf.reduce_sum(self.pro)

        #unstack to get dim=1 pro
        _, p_ = tf.unstack(self.pro, axis = 3)
        sum_1 = tf.reduce_sum(p_ * tf.cast(tf.reduce_min(d_matrix, 3), dtype=tf.float32))
        term_1 = (1 / (p_est_pts + self.eps) * sum_1 )
        
        d_div_p = tf.reduce_min(((d_matrix + self.eps) / (tf.pow(p_, self.alpha) + self.eps)), (1,2))
        d_div_p = tf.minimum(d_div_p, self.max_dist)
        d_div_p = tf.maximum(d_div_p, 0)
        term_2 = tf.reduce_mean(d_div_p)

        self.h_loss = tf.add(term_1, term_2)
        pdb.set_trace()

    #5. evaluation
    def accuracy(self):
        
        #Part 1.Number of correct prediction of label 1.
        sz = [self.cur_batch_size, cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.int64)
        #tensor of correct prediction label 0 and 1
        pred_p_c = tf.where(tf.equal(self.annotations, self.pred_annotation), comp, 1-comp)
        #tensor of correct prediction label 1
        comp2 = comp * 2
        pred_p1_c = tf.where(tf.equal(tf.add(self.annotations, pred_p_c), comp2), comp, 1-comp)
        #number of correct prediction label 1
        self.pred_p_c_num = tf.reduce_sum(pred_p1_c, name='pred_p1_num_c')
        self.pred_p01_c_num = tf.reduce_sum(pred_p_c)
      

        #Part 2.Number of prediction label 1
        self.pred_p_num = tf.reduce_sum(self.pred_annotation)

        #Part 3.Number of label 1 in annotaions
        self.anno_num = tf.reduce_sum(self.annotations)
        
        #IOU accuracy
        self.accu_iou_tensor = (self.pred_p_c_num) / (self.pred_p_num + self.anno_num - self.pred_p_c_num) * 100
        #pixel accuracy
        self.accu_tensor = self.pred_p_c_num / self.anno_num * 100

    def accuracy_lower(self):
        
        #Part 1.Number of correct prediction of label 1.
        sz = [self.cur_batch_size, cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.int64)
        self.pred_anno_lower = tf.cast(self.pred_anno_lower, dtype=tf.int64)
        #tensor of correct prediction label 0 and 1
        pred_p_c = tf.where(tf.equal(self.annotations, self.pred_anno_lower), comp, 1-comp)
        #tensor of correct prediction label 1
        comp2 = comp * 2
        pred_p1_c = tf.where(tf.equal(tf.add(self.annotations, pred_p_c), comp2), comp, 1-comp)
        #number of correct prediction label 1
        self.pred_p_c_num_lower = tf.reduce_sum(pred_p1_c, name='pred_p1_num_c')
        self.pred_p01_c_num_lower = tf.reduce_sum(pred_p_c)
      

        #Part 2.Number of prediction label 1
        self.pred_p_num_lower = tf.reduce_sum(self.pred_anno_lower)

        #Part 3.Number of label 1 in annotaions
        self.anno_num_lower = tf.reduce_sum(self.annotations)
        
        #IOU accuracy
        self.accu_iou_tensor_lower = (self.pred_p_c_num_lower) / (self.pred_p_num_lower + self.anno_num_lower - self.pred_p_c_num_lower) * 100
        #pixel accuracy
        self.accu_tensor_lower = self.pred_p_c_num_lower / self.anno_num_lower * 100

     
        
    def calculate_acc(self, pred_anno, anno):
        with tf.name_scope('accu'):
            self.accu_iou, self.accu = accu.caculate_accurary(pred_anno, anno)
    
    #eval_class_label
    def eval(self):
        with tf.name_scope('eval'):
            is_correct = tf.equal(tf.squeeze(tf.cast(self.pred_label, tf.int32), axis=1), self.class_labels)
            sum_ = tf.cast(tf.reduce_sum(tf.cast(is_correct, tf.int32)), tf.float32)
            self.acc_label = tf.multiply(sum_, 100/(self.batch_size))

    #7. summary
    def summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('train_loss', self.loss)
            #tf.summary.scalar('accu', self.accu)
            #tf.summary.scalar('iou_accu', self.accu_iou)
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.summary_op = tf.summary.merge_all()
    
    #8. graph build
    def build(self):
        pass
    def build_vis(self):
        pass
    def build_video(self):
        pass


    #9. update lr
    def try_update_lr(self):
        try:
            with open(cfgs.learning_rate_path) as f:
                lr_ = float(f.readline().split('\n')[0])
                if self.learning_rate != lr_:
                    self.learning_rate = lr_
                    print('learning rate change from to %g' % self.learning_rate)
        except:
            pass

    #10. Model recover
    def recover_model(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restore finished')
        
        return saver
    
    #11. train or valid once 
    def create_re_dir(self):
        self.re_save_dir="%s%s" % (self.result_dir, datetime.datetime.now())

        self.re_save_dir_im = os.path.join(self.re_save_dir, 'images')
        self.re_save_dir_heat = os.path.join(self.re_save_dir, 'heatmap')
        self.re_save_dir_ellip = os.path.join(self.re_save_dir, 'ellip')
        self.re_save_dir_transheat = os.path.join(self.re_save_dir, 'transheat')
        if not os.path.exists(self.re_save_dir_im):
            os.makedirs(self.re_save_dir_im)
        if not os.path.exists(self.re_save_dir_heat):
            os.makedirs(self.re_save_dir_heat)
        if not os.path.exists(self.re_save_dir_ellip):
            os.makedirs(self.re_save_dir_ellip)
        if not os.path.exists(self.re_save_dir_transheat):
            os.makedirs(self.re_save_dir_transheat)

    def vis_one_im(self):
       if cfgs.anno:
           im_ = pred_visualize(self.vis_image.copy(), self.vis_pred).astype(np.uint8)
           #im_ = pred_visualize(self.vis_image.copy(), self.vis_anno).astype(np.uint8)
           utils.save_image(im_, self.re_save_dir_im, name='inp_' + self.filename + '.jpg')
       if cfgs.fit_ellip:
           im_ellip = fit_ellipse_findContours(self.vis_image.copy(), np.expand_dims(self.vis_pred, axis=2).astype(np.uint8))
           utils.save_image(im_ellip, self.re_save_dir_ellip, name='ellip_' + self.filename + '.jpg')
       if cfgs.heatmap:
           heat_map = density_heatmap(self.vis_pred_prob[:, :, 1])
           utils.save_image(heat_map, self.re_save_dir_heat, name='heat_' + self.filename + '.jpg')
       if cfgs.trans_heat and cfgs.heatmap:
           trans_heat_map = translucent_heatmap(self.vis_image.copy(), heat_map.astype(np.uint8).copy())
           utils.save_image(trans_heat_map, self.re_save_dir_transheat, name='trans_heat_' + self.filenaem + '.jpg')

    #Visualize the result
    def visualize(self, sess):
        pass

    #Visualize the video result
    def vis_video_once(self, sess):
        pass

    #Evaluate all validation dataset once 
    def valid_once(self, sess, writer, epoch, step):

        pass
    
    
    def train_one_epoch(self, sess, writer, epoch, step):
        print('base_train_one_epoch')
        pass

    def train(self):
        #Check if has the log file
        if not os.path.exists(self.logs_dir):
            print("The logs path '%s' is not found" % self.logs_dir)
            print("Create now..")
            os.makedirs(self.logs_dir)
            print("%s is created successfully!" % self.logs_dir)

        print('prepare to train...')
        writer = tf.summary.FileWriter(logdir=self.logs_dir, graph=self.graph)

        print('The graph path is %s' % self.logs_dir)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        #config = tf.ConfigProto(log_device_placement=False, allow_soft_palcement=True)
        with tf.device('/gpu:0'):
            with tf.Session(config=config) as sess:
                #1. initialize all variables
                sess.run(tf.global_variables_initializer())

                #2. Try to recover model
                saver = self.recover_model(sess)
                step = self.current_itr_var.eval()
                cur_epoch = self.cur_epoch.eval() + 1
                print(self.current_itr_var.eval())

                #3. start to train 
                for epoch in range(cur_epoch, self.max_epochs + 1):
                    
                    #3.1 try to change learning rate
                    self.try_update_lr()
                    if epoch != 0 and epoch % 20 == 0:
                        self.learning_rate /= 10
                        pass

                    #3.2 train one epoch
                    #step = self.train_one_epoch(sess, writer, epoch, step)

                    #3.3 save model
                    self.valid_once(sess, writer, epoch, step)
                    self.cur_epoch.load(epoch, sess)
                    self.current_itr_var.load(step, sess)
                    #saver.save(sess, self.logs_dir + 'model.ckpt', step)

        writer.close()

    def vis(self):
        if not os.path.exists(self.logs_dir):
            raise Exception('The logs path %s is not found!' % self.logs_dir)

        print('The logs path is %s.' % self.logs_dir)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.device('/gpu:0'):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                saver = self.recover_model(sess)

                self.visualize(sess)


    def vis_video(self):
        if not os.path.exists(self.logs_dir):
            raise Exception('The logs path %s is not found!' % self.logs_dir)

        print('The logs path is %s.' % self.logs_dir)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.device('/gpu:0'):
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                saver = self.recover_model(sess)

                self.vis_video_once(sess)


