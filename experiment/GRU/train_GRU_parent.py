#This is the base class FCN for sequence 
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
from tools.label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from tools.generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil
#Pretrain model
from tensorflow.python.framework import ops
from DataLoader_corean import DataLoader_c
from models.stgru import STGRU
#from models.stgru_slow_change import STGRU
#from models.stgru_show import STGRU
from models.flownet2 import Flownet2


try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class U_Net(object):

    def __init__(self, mode, max_epochs, batch_size, n_classes, im_sz, init_lr, keep_prob):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.NUM_OF_CLASSESS = n_classes
        self.IMAGE_SIZE = im_sz
        self.graph = tf.get_default_graph()
        self.lr= tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.learning_rate = float(init_lr)
        self.mode = mode
        self.current_itr_var = tf.Variable(0, dtype=tf.int32, name='current_itr', trainable=True)
        self.cur_epoch = tf.Variable(1, dtype=tf.int32, name='cur_epoch', trainable=False)
        #self.cur_batch_size = tf.placeholder(dtype=tf.int32, name='cur_batch_size')

       
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.input_keep_prob = tf.placeholder(tf.float32, shape=[], name='input_keep_prob')
        
        self.at = cfgs.at
        self.gamma = cfgs.gamma

        self.Pre_Net()

        #Hausdorff distance
        self.pos_m = tf.placeholder(tf.int32, shape=[None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 1, 2], name='position_matrix')
        self.gt_key = tf.placeholder(tf.int32, shape=[None, 1, 1, None, 2], name='gt_key_point')
        self.max_dist = math.sqrt(self.IMAGE_SIZE[0]**2 + self.IMAGE_SIZE[1]**2)
        self.eps = cfgs.eps
        self.alpha = cfgs.alpha

    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            #train data loader
            self.train_dl = DataLoader_c(cfgs.IMAGE_SIZE, cfgs.nbr_frames, cfgs.train_anno_path, cfgs.image_path)
            #valid data loader
            self.valid_dl = DataLoader_c(cfgs.IMAGE_SIZE, cfgs.nbr_frames, cfgs.val_anno_path, cfgs.image_path)

    #2. Net
    def Pre_Net(self):

        '''
        Prepare all net models.
        '''
        #1. Load bilinear warping model.
        self.bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
        @ops.RegisterGradient("BilinearWarping")
        def _BilinearWarping(op, grad):
            return self.bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])


        #2. Init flownet
        with tf.variable_scope('flow'):
            self.flow_network = Flownet2(self.bilinear_warping_module)
            self.flow_img0 = tf.placeholder(tf.float32)
            self.flow_img1 = tf.placeholder(tf.float32)
            self.flow_tensor = self.flow_network(self.flow_img0, self.flow_img1, flip=True)
            self.warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.flow_tensor)

        #3. init U-Net for static frame segmentation
        #with tf.variable_scope('seg_static'):
        self.u_net = utils_layers.U_Net_gp()
        self.unet_images = tf.placeholder(tf.float32, shape=[None, cfgs.IMAGE_SIZE[0]+4, cfgs.IMAGE_SIZE[1], cfgs.seq_num+cfgs.cur_channel], name='input_image')
        self.unet_infer_name = 'inference'
        self.unet_ch = cfgs.cur_channel + cfgs.seq_num
        self.unet_keep_pro = cfgs.keep_prob
        #self.class_labels = tf.placeholder(tf.int32, shape=[None], name='class_occlusion_label')
        self.static_output, self.static_anno_pred = self.u_net_inference(self.unet_images, self.unet_infer_name, self.unet_ch, self.unet_keep_pro)

        #4. Init RNN net
        
        self.RNN = STGRU([self.NUM_OF_CLASSESS, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]], [7, 7], self.bilinear_warping_module)
        #self.GRU_op_show()
        self.GRU_op()
        
    def u_net_inference(self, images, inference_name, channel, keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up resnet101 initialized conv layers ...")
        #mean_pixel = np.mean(mean, axis=(0, 1))

        processed_images = utils.process_image(images, cfgs.mean_pixel)

        #processed_images = tf.nn.dropout(processed_images, self.input_keep_prob)

        with tf.variable_scope(inference_name):
            
            #U-Net
            logits = self.u_net.u_net_op(x=processed_images, 
                                         keep_prob_=keep_prob, 
                                         channels=channel,
                                         n_class = cfgs.n_class,
                                         layers = cfgs.layers,
   
                                         features_root=cfgs.features_root,
                                         filter_size = cfgs.filter_size,
                                         pool_size = cfgs.pool_size)
            self.unet_pro = tf.nn.softmax(logits)[:, 2:cfgs.IMAGE_SIZE[0]+2, :, :]
            anno_pred = tf.argmax(tf.nn.softmax(logits), dimension=3, name="prediction")
            #pdb.set_trace()
            print('logits shape', logits.shape)

        return logits[:, 2:cfgs.IMAGE_SIZE[0]+2, :, :], anno_pred[0, 2:cfgs.IMAGE_SIZE[0]+2, :]

    def GRU_op(self):
        
        self.gru_opt, self.gru_loss, self.gru_pred, self.gru_pred_pro, self.gru_lr,\
        self.gru_input_images_tensor, self.gru_input_flow_tensor,\
        self.gru_input_seg_tensor, self.gru_targets = self.RNN.get_optimizer(cfgs.seq_frames)
        #= self.RNN.get_optimizer_slow_change(cfgs.seq_frames)
        #Expand dims
        self.gru_pred = tf.expand_dims(tf.expand_dims(self.gru_pred, axis=0), axis=-1)
        self.gru_targets_acc = tf.expand_dims(tf.expand_dims(self.gru_targets, axis=0), axis=-1)
        self.gru_pred_pro = tf.expand_dims(self.gru_pred_pro, axis=0)

        self.unary_grad_op = tf.gradients(self.gru_loss, self.gru_input_seg_tensor)

    def GRU_op_show(self):
        
        self.gru_opt, self.gru_loss, self.gru_pred, self.gru_pred_pro, self.gru_lr,\
        self.gru_input_images_tensor, self.gru_input_flow_tensor,\
        self.gru_input_seg_tensor, self.gru_targets,\
        self.gru_prev_warped, self.gru_I_diff, self.gru_h_prev_warped = self.RNN.get_optimizer(cfgs.seq_frames)
        #Expand dims
        self.gru_pred = tf.expand_dims(tf.expand_dims(self.gru_pred, axis=0), axis=-1)
        self.gru_targets_acc = tf.expand_dims(tf.expand_dims(self.gru_targets, axis=0), axis=-1)
        self.gru_pred_pro = tf.expand_dims(self.gru_pred_pro, axis=0)
        self.gru_prev_warped = tf.cast(self.gru_prev_warped, tf.int32)
        self.gru_pro_prev_warped = tf.nn.softmax(self.gru_h_prev_warped)
        self.gru_anno_prev_warped = tf.argmax(tf.nn.softmax(self.gru_h_prev_warped), 3)
        self.gru_I_diff = tf.abs(tf.cast(self.gru_I_diff, tf.int32))
        self.unary_grad_op = tf.gradients(self.gru_loss, self.gru_input_seg_tensor)
        

        
    #5. evaluation
    def accuracy(self):
        
        #Part 1.Number of correct prediction of label 1.
        sz = [self.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.int64)
        #tensor of correct prediction label 0 and 1
        pred_p_c = tf.where(tf.equal(self.gru_targets_acc, self.gru_pred), comp, 1-comp)
        #tensor of correct prediction label 1
        comp2 = comp * 2
        pred_p1_c = tf.where(tf.equal(tf.add(self.gru_targets_acc, pred_p_c), comp2), comp, 1-comp)
        #number of correct prediction label 1
        self.pred_p_c_num = tf.reduce_sum(pred_p1_c, name='pred_p1_num_c')
        self.pred_p01_c_num = tf.reduce_sum(pred_p_c)
      

        #Part 2.Number of prediction label 1
        self.pred_p_num = tf.reduce_sum(self.gru_pred)

        #Part 3.Number of label 1 in annotaions
        self.anno_num = tf.reduce_sum(self.gru_targets_acc)
        
        #IOU accuracy
        self.accu_iou_tensor = (self.pred_p_c_num) / (self.pred_p_num + self.anno_num - self.pred_p_c_num) * 100
        #pixel accuracy
        self.accu_tensor = self.pred_p_c_num / self.anno_num * 100

    def accuracy_lower(self):
        
        #Part 1.Number of correct prediction of label 1.
        sz = [self.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 1]
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
    
    #accuracy for label2
    def acc_label2(self):
        '''
        Only caculate accuracy for label 2, the cornea.
        label 0: background
        label 1: instrument
        label 2: cornea
        '''
        
        #Part 1.Number of correct prediction of label 1.
        sz = [self.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.int64)
        #tensor of correct prediction label 2
        comp4 = comp * 4
        self.add4 = tf.add(self.annotations, self.pred_annotation)
        pred_p2_c = tf.where(tf.equal(tf.add(self.annotations, self.pred_annotation), comp4), comp, 1-comp)
        self.pred_p2_c = pred_p2_c
        #Numner of correct prediction of label 2
        self.pred_p2_c_num = tf.reduce_sum(pred_p2_c, name='pred_p2_num_c')


        #Part 2:Number of prediction of label 2
        comp2 = comp * 2
        pred_p2 = tf.where(tf.equal(self.pred_annotation, comp2), comp, 1-comp)
        self.pred_p2_num = tf.reduce_sum(pred_p2)

        #Part 3:Number of annotation of label2
        anno_p2 = tf.where(tf.equal(self.annotations, comp2), comp, 1-comp)
        self.anno2_num = tf.reduce_sum(anno_p2)

        #IOU accuracy
        self.accu_iou_tensor = (self.pred_p2_c_num) / (self.pred_p2_num + self.anno2_num - self.pred_p2_c_num) * 100
        #pixel accuracy
        self.accu_tensor = self.pred_p2_c_num / self.anno2_num * 100

    def acc_label2_lower(self):
        '''
        Only caculate accuracy for label 2, the cornea.
        label 0: background
        label 1: instrument
        label 2: cornea
        '''
        
        #Part 1.Number of correct prediction of label 1.
        sz = [self.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.int64)
        self.pred_anno_lower = tf.cast(self.pred_anno_lower, dtype=tf.int64)
        #tensor of correct prediction label 2
        comp4 = comp * 4
        pred_p2_c = tf.where(tf.equal(tf.add(self.annotations, self.pred_anno_lower), comp4), comp, 1-comp)
        #Numner of correct prediction of label 2
        self.pred_p2_c_num_lower = tf.reduce_sum(pred_p2_c, name='pred_p2_num_c')


        #Part 2:Number of prediction of label 2
        comp2 = comp * 2
        pred_p2 = tf.where(tf.equal(self.pred_anno_lower, comp2), comp, 1-comp)
        self.pred_p2_num_lower = tf.reduce_sum(pred_p2)

        #Part 3:Number of annotation of label2
        anno_p2 = tf.where(tf.equal(self.annotations, comp2), comp, 1-comp)
        self.anno2_num_lower = tf.reduce_sum(anno_p2)

        #IOU accuracy
        self.accu_iou_tensor_lower = (self.pred_p2_c_num_lower) / (self.pred_p2_num_lower + self.anno2_num_lower - self.pred_p2_c_num_lower) * 100
        #pixel accuracy
        self.accu_tensor_lower = self.pred_p2_c_num_lower / self.anno2_num_lower * 100

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
    def return_saver_ckpt(self, sess, logs_dir, var_list):
    
        saver = tf.train.Saver(var_list)
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model %s restore finished' % logs_dir)

        return saver
        
    def return_saver(self, sess, logs_dir, model_name, var_list):
        
        saver = tf.train.Saver(var_list)
        if os.path.exists(logs_dir):
            saver.restore(sess, os.path.join(logs_dir, model_name))
            print('Model %s restore finished' % logs_dir)

        return saver
     
    def recover_model(self, sess):
        var_list = tf.trainable_variables()
        
        var_not_flow = [k for k in var_list if not k.name.startswith('flow')]

        var_static = [k for k in var_not_flow if k.name.startswith('inference')]
        var_gru = [k for k in var_not_flow if not k.name.startswith('inference')]
        var_flow = [k for k in var_list if k.name.startswith('flow')]
        
        loader_static = self.return_saver_ckpt(sess, cfgs.unet_logs_dir, var_static)
        loader_flow = self.return_saver(sess, cfgs.flow_logs_dir, cfgs.flow_logs_name, var_flow)
        saver = self.return_saver_ckpt(sess, cfgs.gru_logs_dir, var_gru)

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
        if not os.path.exists(cfgs.gru_logs_dir):
            print("The logs path '%s' is not found" % cfgs.gru_logs_dir)
            print("Create now..")
            os.makedirs(cfgs.gru_logs_dir)
            print("%s is created successfully!" % cfgs.gru_logs_dir)

        print('prepare to train...')
        #writer = tf.summary.FileWriter(logdir=self.logs_dir, graph=self.graph)

        print('The graph path is %s' % cfgs.gru_logs_dir)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
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
                    #step = self.train_one_epoch(sess, self.train_dl, cfgs.train_num, epoch, step)
                    #for train_sub_flow
                    #step = self.train_one_epoch_remv_occ(sess, self.train_dl, cfgs.train_num, epoch, step)
                    #step = self.train_one_epoch_remv_occ_segm(sess, self.train_dl, cfgs.train_num, epoch, step)
                    step = self.train_one_epoch_warp_inpt_flow(sess, self.train_dl, cfgs.train_num, epoch, step)

                    #self.warp_one_im(sess, step)
                    
                    #3.3 save model
                    self.valid_once(sess, self.valid_dl, cfgs.valid_num, epoch, step)
                    self.cur_epoch.load(epoch, sess)
                    self.current_itr_var.load(step, sess)
                    saver.save(sess, cfgs.gru_logs_dir + 'model.ckpt', step)

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


