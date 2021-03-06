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
from tools.config import Config
import shutil
#Pretrain model
from tensorflow.python.framework import ops
from DataLoader_corean_inpt_da_reconst_seq import DataLoader_c
#from DataLoader_corean_inpt import DataLoader_c
from models.stgru import STGRU
from models.flownet2 import Flownet2
from models.inpaint_model import InpaintModel
import tools.data_preprocess_tf as dp_tf


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
        self.cur_epoch = tf.Variable(1, dtype=tf.int32, name='cur_epoch', trainable=True)
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
            self.train_dl = DataLoader_c()
            #valid data loader
            self.valid_dl = DataLoader_c()

    #2. Net
    def Pre_Net(self):

        '''
        Prepare all net models.
        '''
        # Input
        sz = [1, cfgs.U_IMAGE_SIZE[0], cfgs.U_IMAGE_SIZE[1], 3]
        sz1 = [1, cfgs.U_IMAGE_SIZE[0], cfgs.U_IMAGE_SIZE[1], 1]
        b_sz = [cfgs.U_IMAGE_SIZE[0], cfgs.U_IMAGE_SIZE[1]]
        re_sz = [1, cfgs.inpt_resize_im_sz[0], cfgs.inpt_resize_im_sz[1], 3]
        b_re_sz = [cfgs.inpt_resize_im_sz[0], cfgs.inpt_resize_im_sz[1]]
        re_sz2 = [1, cfgs.inpt_resize_im_sz[0], cfgs.inpt_resize_im_sz[1], 2]
        re_sz1 = [1, cfgs.inpt_resize_im_sz[0], cfgs.inpt_resize_im_sz[1], 1]

        comp = tf.ones(re_sz2, dtype=tf.float32)
        comp_ori = tf.ones(sz1, dtype=tf.float32)

        self.prev_img = tf.placeholder(tf.float32, shape=sz)
        self.cur_img = tf.placeholder(tf.float32, shape=sz)
        self.cur_mask = tf.placeholder(tf.float32, shape=re_sz)

        self.re_prev_img = tf.image.resize_bicubic(self.prev_img, b_re_sz)
        self.re_cur_img = tf.image.resize_bicubic(self.cur_img, b_re_sz)


        #1. Load bilinear warping model.
        self.bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
        @ops.RegisterGradient("BilinearWarping")
        def _BilinearWarping(op, grad):
            return self.bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])
        
        #2. Pad for U-Net
        paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
        #self.pad_warped_prev_im = tf.pad(self.warped_prev_im, paddings, 'CONSTANT')
        self.pad_cur_im = tf.pad(self.cur_img, paddings, 'CONSTANT')

        
        #3. current frame segmetation
        self.u_net = utils_layers.U_Net_gp()
        self.unet_infer_name = 'inference'
        self.unet_ch = cfgs.cur_channel + cfgs.seq_num
        self.unet_keep_pro = cfgs.keep_prob
        self.cur_static_output, self.cur_static_anno_pred = self.u_net_inference(self.pad_cur_im, self.unet_infer_name, self.unet_ch, self.unet_keep_pro)
        

        #4. Init flownet
        with tf.variable_scope('flow'):
            self.flow_network = Flownet2(self.bilinear_warping_module)
            self.flow_tensor = self.flow_network(self.re_cur_img, self.re_prev_img, flip=True)
        
        #5. Normal flow data
        inst_mask = tf.where(tf.equal(tf.expand_dims(self.cur_static_anno_pred[:, :, :, 1], axis=3), 1), comp_ori, 1-comp_ori)
        self.re_cur_inst_mask = tf.image.resize_bicubic(inst_mask*255, b_re_sz)
        mask1 = tf.cast(self.re_cur_inst_mask > 127, tf.float32)
        # max pooling
        mask1 = tf.nn.max_pool(mask1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        mask1 = tf.nn.max_pool(mask1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        mask1 = tf.nn.max_pool(mask1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        mask1 = tf.nn.max_pool(mask1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        mask1 = tf.nn.max_pool(mask1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')



        mask = tf.cast(tf.concat([mask1, mask1], axis=3), tf.float32)
        remove_mask = tf.multiply((comp-mask), self.flow_tensor)
        self.cur_mask = mask * 255
        self.inpt_flow_input, self.max_v = dp_tf.normal_data(remove_mask)
        self.inpt_data = dp_tf.concat_data(self.inpt_flow_input, self.cur_mask)
    
        
        #6. Init flow-inpaint net
        config = Config('cfgs/inpaint.yml')
        self.inpt_network = InpaintModel()
        self.inpt_g_vars, self.inpt_pred_flow, self.pred_complete_flow = self.inpt_network.build_graph(
            self.inpt_data, config=config)
        self.inpt_pred_flow = self.inpt_pred_flow * self.max_v
        #self.inpt_pred_flow_re = tf.image.resize_bicubic(self.inpt_pred_flow, b_sz)
        self.inpt_pred_flow_re = tf.multiply(self.flow_tensor, comp-mask1)
        self.warped_prev_im = self.bilinear_warping_module.bilinear_warping(self.re_prev_img, self.inpt_pred_flow)
        self.warped_prev_im = tf.image.resize_bicubic(self.warped_prev_im, b_sz)

        #7. Pad for U-Net
        self.pad_warped_prev_im = tf.pad(self.warped_prev_im, paddings, 'CONSTANT')

        #8. prev_frame after warped segmentation
        self.warped_static_output, self.warped_static_anno_pred = self.u_net_inference(self.pad_warped_prev_im, self.unet_infer_name, self.unet_ch, self.unet_keep_pro)
        
        #9. Reconstruction
        u_mask = tf.image.resize_bicubic(self.cur_mask, b_sz)
        self.u_mask = tf.cast(u_mask[0:1, :, :, 0:1] > 127.5, tf.float32)
        insect_area = tf.cast((tf.multiply(self.warped_static_anno_pred, tf.cast(self.u_mask, tf.int64))) / 2, tf.int32)
        #self.reconst_cur_anno = tf.cast(self.cur_static_anno_pred / 2, tf.int32) + insect_area
        self.reconst_cur_anno = tf.cast(self.cur_static_anno_pred / 2, tf.int32) + tf.cast(self.warped_static_anno_pred / 2, tf.int32)






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
        anno_pred = tf.expand_dims(anno_pred, axis=3)
        anno_pred = tf.concat([anno_pred, anno_pred, anno_pred], 3)    

        return logits[:, 1:cfgs.U_IMAGE_SIZE[0]+1, :, :], anno_pred[:, 1:cfgs.U_IMAGE_SIZE[0]+1, :, :]
        #return logits, anno_pred

  


    def loss(self):
        
        #self.inpt_flow = tf.placeholder(tf.float32)
        self.inpt_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.inpt_pred_flow)
        self.loss = tf.reduce_mean(tf.abs(self.inpt_warped_im - self.flow_img0))

    def train_optimizer(self):

        
        optimizer = tf.train.AdamOptimizer(cfgs.inpt_lr, beta1=0.5, beta2=0.9)
        #optimizer = tf.train.AdamOptimizer(cfgs.inpt_lr)

        var_list = tf.trainable_variables()

        var_not_flow = [k for k in var_list if not k.name.startswith('flow')]

        var_inpt = [k for k in var_list if k.name.startswith('inpaint_net')]
        var_flow = [k for k in var_list if k.name.startswith('flow')]

        grads = optimizer.compute_gradients(self.loss, var_list=var_inpt)
        self.opt = optimizer.apply_gradients(grads)
               
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
        var_inpt = [k for k in var_list if k.name.startswith('inpaint_net')]
        var_flow = [k for k in var_list if k.name.startswith('flow')]
        
        var_save = [k for k in var_not_flow if not k.name.startswith('inference')]

        
        loader_static = self.return_saver_ckpt(sess, cfgs.unet_logs_dir, var_static)
        loader_flow = self.return_saver(sess, cfgs.flow_logs_dir, cfgs.flow_logs_name, var_flow)
        saver = self.return_saver_ckpt(sess, cfgs.gru_logs_dir, var_save)

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

                    
                    #3.3 save model
                    self.valid_one_epoch_warped_im(sess, self.valid_dl, cfgs.valid_num, epoch, step)
                    self.cur_epoch.load(epoch, sess)
                    self.current_itr_var.load(step, sess)
                    #saver.save(sess, cfgs.gru_logs_dir + 'model.ckpt', step)


    
