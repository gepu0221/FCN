#This code is for training, adding ellipse center to the loss function.
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import TensorflowUtils as utils
#import read_data_finegrain as scene_parsing_fg
import read_data as scene_parsing
import datetime
import pdb
from BatchReader_multi_ellip_label2 import *
import CaculateAccurary_rect as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

#from train_seq_parent import FCNNet
from train_Unet_parent import U_Net as FCNNet

try:
    #from .cfgs.config_train_resnet_fg import cfgs
    from .cfgs.config_train_u_net import cfgs
except Exception:
    #from cfgs.config_train_resnet_fg import cfgs
    from cfgs.config_train_u_net import cfgs


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class SeqFCNNet(FCNNet):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir):

        FCNNet.__init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir)
        
        #mask
        self.seq_num = cfgs.seq_num
        self.cur_channel = cfgs.cur_channel
        self.channel = self.cur_channel + self.seq_num
        self.inference_name = 'inference'
        self.images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], cfgs.seq_num+self.cur_channel], name='input_image')
        self.create_view_path()
        self.coord_map_x, self.coord_map_y = self.generate_coord_map(self.batch_size)
        self.coord_x_tensor = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]], name='coord_x_map_tensor')
        self.coord_y_tensor = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]], name='coord_y_map_tensor')
        
        self.ellip_low = tf.placeholder(tf.float32, shape=[None], name='ellipse_info_lower_axis')
        self.ellip_high = tf.placeholder(tf.float32, shape=[None], name='ellipse_info_higher_axis')
        self.ellip_axis = tf.placeholder(tf.float32, shape=[None], name='ellipse_info_mean_axis')

        accu.create_ellipse_f()
        self.e_acc = accu.Ellip_acc()

    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations,self.train_labels,  self.train_filenames = get_data_cache(self.train_records, self.batch_size, False, 'get_data_train')
            self.valid_images, self.valid_cur_ims, self.valid_ellip_infos, self.valid_annotations, self.valid_labels, self.valid_filenames = get_data_cache(self.valid_records, self.batch_size, False, 'get_data_valid_mask')


    def get_data_vis(self):
        with tf.device('/cpu:0'):
            self.vis_images, self.vis_cur_images, self.vis_ellip_infos, self.vis_annotations, self.vis_filenames, self.vis_init = get_data_vis(self.valid_records, self.batch_size)


    def get_data_video(self):
        with tf.device('/cpu:0'):
            self.video_images, self.video_cur_ims, self.video_filenames, self.video_init = get_data_video(self.valid_records, self.batch_size)

    #2. loss 
    def generate_coord_map(self, cur_batch_sz):
        
        coord_map_x = np.zeros((cur_batch_sz, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]))
        coord_map_y = np.zeros((cur_batch_sz, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]))
        #pdb.set_trace()
        for i in range(cur_batch_sz):
            for j in range(cfgs.IMAGE_SIZE[0]):
                for k in range(cfgs.IMAGE_SIZE[1]):
                    coord_map_x[i][j][k] = j
                    coord_map_y[i][j][k] = k

        #pdb.set_trace()    
        return coord_map_x, coord_map_y


    
    def center_wh_range_global_loss(self):
        
        #prepare
        sz = [self.cur_batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]]
        comp = tf.ones(sz, dtype=tf.float32)
        zero_e = tf.zeros([1, cfgs.batch_size], dtype=tf.float32)
        zero2_e = tf.zeros(sz, dtype=tf.float32)
        
        pred_sum = tf.cast(tf.reduce_sum(self.pro[:, :, :, 1], [1, 2]), dtype=tf.float32)
        pred_x = tf.multiply(self.pro[:, :, :, 1], self.coord_x_tensor)
        pred_y = tf.multiply(self.pro[:, :, :, 1], self.coord_y_tensor)
        self.pred_cx = tf.reduce_sum(pred_x, [1, 2]) / pred_sum
        self.pred_cy = tf.reduce_sum(pred_y, [1, 2]) / pred_sum
       

        anno_sum = tf.cast(tf.reduce_sum(tf.squeeze(self.annotations, squeeze_dims=[3]), [1, 2]), dtype=tf.float32)
        anno_x =  tf.multiply(tf.cast(tf.squeeze(self.annotations, squeeze_dims=[3]), dtype=tf.float32), self.coord_x_tensor)
        anno_y =  tf.multiply(tf.cast(tf.squeeze(self.annotations, squeeze_dims=[3]), dtype=tf.float32), self.coord_y_tensor)
        self.anno_cx = tf.reduce_sum(anno_x, [1, 2]) / anno_sum
        self.anno_cy = tf.reduce_sum(anno_y, [1, 2]) / anno_sum

        self.center_loss = tf.reduce_mean(tf.pow((self.pred_cx - self.anno_cx), 2) + tf.pow((self.pred_cy - self.anno_cy), 2))
        #--------------------
        #pred_cx_m = tf.expand_dims(self.pred_cx, 0)
        pred_cx_m = tf.expand_dims(self.anno_cx, 0)
        for i in range(cfgs.batch_size-1):
            pred_cx_m = tf.concat([pred_cx_m, zero_e], 0)
        comp_cx = tf.matmul(pred_cx_m, tf.reshape(comp, [cfgs.batch_size, cfgs.IMAGE_SIZE[0]*cfgs.IMAGE_SIZE[1]]), transpose_a=True)
        comp_cx = tf.reshape(comp_cx, [self.cur_batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]])
        #-------------------
       
        #pred_cy_m = tf.expand_dims(self.pred_cy, 0)
        pred_cy_m = tf.expand_dims(self.anno_cy, 0)
        for i in range(cfgs.batch_size-1):
            pred_cy_m = tf.concat([pred_cy_m, zero_e], 0)
        comp_cy = tf.matmul(pred_cy_m, tf.reshape(comp, [cfgs.batch_size, cfgs.IMAGE_SIZE[0]*cfgs.IMAGE_SIZE[1]]), transpose_a=True)
        comp_cy = tf.reshape(comp_cy, [self.cur_batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]])
        #self.pred_dis_map = tf.multiply(tf.pow((comp_cx - self.coord_x_tensor), 2), self.pro[:, :, :, 1]) + tf.multiply(tf.pow((comp_cy- self.coord_y_tensor), 2), self.pro[:, :, :, 1])
        self.pred_dis_map = tf.pow((comp_cx - self.coord_x_tensor), 2) + tf.pow((comp_cy- self.coord_y_tensor), 2)
        #Lower axis
        #-------------------------
        anno_axis_m = tf.expand_dims(self.ellip_axis, 0)
        for i in range(cfgs.batch_size-1):
            anno_axis_m = tf.concat([anno_axis_m, zero_e], 0)
        anno_axis_comp = tf.matmul(anno_axis_m, tf.reshape(comp, [cfgs.batch_size, cfgs.IMAGE_SIZE[0]*cfgs.IMAGE_SIZE[1]]), transpose_a=True)
        anno_axis_comp = tf.reshape(anno_axis_comp, [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]])
        anno_axis_comp = tf.pow(anno_axis_comp, 2)
        
        offset_ = tf.multiply((tf.abs(anno_axis_comp - self.pred_dis_map)), self.pro[:, :, :, 1])
       
        #-----------------------------------------------
        self.wh_loss = tf.reduce_mean(offset_)


    def loss(self):

        self.logits, self.class_logits = self.inference(self.images, self.inference_name, self.channel, self.keep_prob)

        #1. U-net
        self.logits = self.logits[:, 2:cfgs.ANNO_IMAGE_SIZE[0]+2, :, :]
        self.pro = tf.nn.softmax(self.logits)
        self.pred_annotation = tf.expand_dims(tf.argmax(self.pro, dimension=3, name='pred'), dim=3)
        
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                       labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                       name='entropy_loss')))


        

        sz = [self.cur_batch_size, cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1]]
        im_comp = tf.ones(sz, dtype=tf.int32)
        self.pred_anno_lower = tf.expand_dims(tf.where(tf.less_equal(self.pro[:, :, :, 2], cfgs.low_pro), 1-im_comp, im_comp), dim=3)
  

    def generate_mask_im(self):
        
        self.mask_ims = tf.multiply(tf.cast(self.pred_anno_lower, dtype=tf.float32), self.images)


    def generate_mask_im_filter_occ(self):
        
        '''
        Generate image mask after Net1 using a lower probility.
        Filter occlusion part which is labeled with 2.
        '''
        
        self.mask_ims_lower = tf.multiply(tf.cast(self.pred_anno_lower, dtype=tf.float32), self.images[:, 2:cfgs.ANNO_IMAGE_SIZE[0]+2, :, :])
        sz = [self.batch_size, cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.int64)
        #Instrument mask filter
        sz = [cfgs.batch_size, cfgs.ANNO_IMAGE_SIZE[0], cfgs.ANNO_IMAGE_SIZE[1]]
        im_comp = tf.ones(sz, dtype=tf.int64)
        self.pred_inst_lower = tf.expand_dims(tf.where(tf.less_equal(self.pro[:, :, :, 1], cfgs.inst_low_pro), 1-im_comp, im_comp), dim=3)

        self.inst_mask = tf.cast(tf.where(tf.equal(self.pred_inst_lower, comp), 1-comp, comp), tf.float32)

        self.mask_ims = tf.multiply(self.mask_ims_lower, self.inst_mask)
        #self.mask_ims = self.mask_ims_lower
        #pdb.set_trace()
        self.mask_anno = tf.multiply(self.pred_anno_lower, tf.cast(self.inst_mask, tf.int32))
        #pdb.set_trace()
    #3. accuracy
    def calculate_acc(self, im, filenames, pred_anno, pred_pro, anno, gt_ellip_info, if_valid=False, if_epoch=True):
        with tf.name_scope('ellip_accu'):
            if cfgs.test_accu and if_epoch:
                #self.accu_iou, self.accu = accu.caculate_accurary(pred_anno, anno)

                #ellipse loss
                #self.ellip_acc = accu.caculate_ellip_accu(im, filenames, pred_anno, pred_pro, gt_ellip_info, if_valid)
                #Hausdorff loss
                self.ellip_acc = self.e_acc.caculate_ellip_accu(im, filenames, pred_anno, pred_pro, gt_ellip_info, if_valid, if_epoch)
            
            else:
                #self.accu_iou = 0
                #self.accu = 0
                self.ellip_acc = 0
    
    #5. build graph
    
    def build(self):
        
        self.get_data_cache()
        self.loss()
        self.generate_mask_im_filter_occ()
        self.accuracy()
        self.accuracy_lower()
        self.train_optimizer()
        self.summary()

    def build_vis(self):
        self.get_data_vis()
        self.loss()

    def build_video(self):
        self.get_data_video()
        self.loss()
    
    def build_h_loss(self):
        self.get_data_cache()
        self.loss()
        self.h_loss()

    #6. else
    def create_view_path(self):
        train_path = os.path.join(cfgs.view_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        valid_path = os.path.join(cfgs.view_path, 'valid')
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)


    def view_one(self, fn, pred_anno, pred_pro, im, step):
        path_ = os.path.join(cfgs.view_path, 'train')
        
        if cfgs.test_view:
            filename = fn.strip().decode('utf-8')
        else:
            filename = str(step)+'_'+fn.strip().decode('utf-8')

        pred_anno_im = (pred_anno*255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_, filename+'_anno.bmp'), pred_anno_im)
        cv2.imwrite(os.path.join(path_, filename+'_im.bmp'), im[:,:,0])
        heatmap = density_heatmap(pred_pro[:,:,1])
        cv2.imwrite(os.path.join(path_, filename+'_heat.bmp'), heatmap)
        if cfgs.view_seq:
            for i in range(cfgs.seq_num):
                im_ = im[:,:,self.cur_channel-1+i]
                cv2.imwrite(os.path.join(path_, filename+'seq_'+str(i+1)+'.bmp'), im_)
            
    def view(self, fns, pred_annos, pred_pros, ims, step):
        num_ = fns.shape[0]
        if cfgs.random_view:
            choosen = random.randint(0, num_-1)
            self.view_one(fns[choosen], pred_annos[choosen], pred_pros[choosen], ims[choosen], step)
        else:
            for i in range(num_):
                self.view_one(fns[i], pred_annos[i], pred_pros[i], ims[i], step)   

    def view_one_valid(self, fn, pred_anno, pred_pro, im, step):
        path_ = os.path.join(cfgs.view_path, 'valid')
        if cfgs.test_view:
            filename = fn.strip().decode('utf-8')
        else:
            filename = str(step)+'_'+fn.strip().decode('utf-8')
        
        pred_anno_im = (pred_anno*255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_, filename+'_anno.bmp'), pred_anno_im)
        cv2.imwrite(os.path.join(path_, filename+'_im.bmp'), im[:,:,0])
        heatmap = density_heatmap(pred_pro[:,:,1])
        cv2.imwrite(os.path.join(path_, filename+'_heat.bmp'), heatmap)
        if cfgs.view_seq:
            for i in range(cfgs.seq_num):
                im_ = im[:,:,self.cur_channel-1+i]
                cv2.imwrite(os.path.join(path_, filename+'seq_'+str(i+1)+'.bmp'), im_)
            
    def view_valid(self, fns, pred_annos, pred_pros, ims, step):
        num_ = fns.shape[0]
        if cfgs.random_view:
            choosen = random.randint(0, num_-1)
            self.view_one_valid(fns[choosen], pred_annos[choosen], pred_pros[choosen], ims[choosen], step)
        else:
            for i in range(num_):
                self.view_one_valid(fns[i], pred_annos[i], pred_pros[i], ims[i], step)   


    #generate image mask
    def im_mask_view_one(self, fn, pred_anno, pred_pro, im, step):
        path_ = os.path.join(cfgs.view_path, 'train')
        
        if cfgs.test_view:
            filename = fn.strip().decode('utf-8')
        else:
            filename = str(step)+'_'+fn.strip().decode('utf-8')

        pred_anno_im = cv2.cvtColor((pred_anno).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_, filename+'.bmp'), pred_anno_im)
        #cv2.imwrite(os.path.join(path_, filename+'_im.bmp'), im[:,:,0])
        #heatmap = density_heatmap(pred_pro[:,:,1])
        #cv2.imwrite(os.path.join(path_, filename+'_heat.bmp'), heatmap)
        if cfgs.view_seq:
            for i in range(cfgs.seq_num):
                im_ = im[:,:,self.cur_channel-1+i]
                cv2.imwrite(os.path.join(path_, filename+'seq_'+str(i+1)+'.bmp'), im_)
            
    def im_mask_view(self, fns, pred_annos, pred_pros, ims, step):
        num_ = fns.shape[0]
        if cfgs.random_view:
            choosen = random.randint(0, num_-1)
            self.im_mask_view_one(fns[choosen], pred_annos[choosen], pred_pros[choosen], ims[choosen], step)
        else:
            for i in range(num_):
                self.im_mask_view_one(fns[i], pred_annos[i], pred_pros[i], ims[i], step)   

    def im_mask_view_one_valid(self, fn, pred_anno, pred_pro, im, step):
        path_ = os.path.join(cfgs.view_path, 'valid')
        if cfgs.test_view:
            filename = fn.strip().decode('utf-8')
        else:
            filename = str(step)+'_'+fn.strip().decode('utf-8')
        
        pred_anno_im = cv2.cvtColor((pred_anno).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_, filename+'.bmp'), pred_anno_im)
        #cv2.imwrite(os.path.join(path_, filename+'_im.bmp'), im[:,:,0])
        #heatmap = density_heatmap(pred_pro[:,:,1])
        #cv2.imwrite(os.path.join(path_, filename+'_heat.bmp'), heatmap)
        if cfgs.view_seq:
            for i in range(cfgs.seq_num):
                im_ = im[:,:,self.cur_channel-1+i]
                cv2.imwrite(os.path.join(path_, filename+'seq_'+str(i+1)+'.bmp'), im_)
            
    def im_mask_view_valid(self, fns, pred_annos, pred_pros, ims, step):
        num_ = fns.shape[0]
        if cfgs.random_view:
            choosen = random.randint(0, num_-1)
            self.im_mask_view_one_valid(fns[choosen], pred_annos[choosen], pred_pros[choosen], ims[choosen], step)
        else:
            for i in range(num_):
                self.im_mask_view_one_valid(fns[i], pred_annos[i], pred_pros[i], ims[i], step)   


    #Evaluate all validation dataset once 
    def valid_once(self, sess, writer, epoch, step):

        count = 0
        sum_acc = 0
        sum_acc_iou = 0
        sum_acc_ellip = 0
        t0 = time.time()
        
        if_epoch = False
        if epoch % 5 == 0:
            if_epoch = True

        try:
            total_loss = 0
            #self.per_e_valid_batch = 2
            while count<self.per_e_valid_batch:
                count +=1
                images_, cur_ims, ellip_infos_, annos_, filenames = sess.run([self.valid_images, self.valid_cur_ims, self.valid_ellip_infos, self.valid_annotations, self.valid_filenames])

                cur_batch_size = images_.shape[0]
                if cur_batch_size == cfgs.batch_size:
                    coord_map_x_cur, coord_map_y_cur = self.coord_map_x, self.coord_map_y
                else:
                    coord_map_x_cur, coord_map_y_cur = self.generate_coord_map(cur_batch_size)
                
                if cur_batch_size == cfgs.batch_size:
                    
                    ellip_info_low = np.min(ellip_infos_[:, 2:], 1) / 2
                    ellip_info_high = np.max(ellip_infos_[:, 2:], 1) / 2
                    ellip_info_mean = np.mean(ellip_infos_[:, 2:], 1) / 4

                    mask_ims, pred_anno_, pred_seq_pro_, summary_str = sess.run(
                    #fetches=[self.pred_annotation, self.pro, self.summary_op, self.loss, self.accu_tensor, self.accu_iou_tensor],
                    #fetches=[self.pred_anno_lower, self.pro, self.summary_op, self.loss, self.accu_tensor_lower, self.accu_iou_tensor_lower],
                    fetches=[self.mask_ims, self.mask_anno, self.pro, self.summary_op],
                    feed_dict={self.images: images_, 
                               self.annotations: annos_, self.lr: self.learning_rate,
                               self.keep_prob: 1,
                               self.input_keep_prob: 1,
                               self.cur_batch_size: cur_batch_size,
                               self.coord_x_tensor: coord_map_x_cur,
                               self.coord_y_tensor: coord_map_y_cur,
                               self.ellip_axis: ellip_info_mean})
                
                    #View result
                    #self.view_valid(filenames, pred_anno, pred_seq_pro, images_, step)
                    #self.im_mask_view_valid(filenames, pred_anno, pred_seq_pro, images_, step)


                    writer.add_summary(summary_str, global_step=step)
                    #self.calculate_acc(annos_, filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_, True, if_epoch)
                    #self.calculate_acc(mask_ims.copy(), filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_, if_epoch=if_epoch)
                    self.calculate_acc(cur_ims[:, 2:cfgs.ANNO_IMAGE_SIZE[0]+2, :, :], filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_, if_epoch=if_epoch)



                    self.accu = 0
                    self.accu_iou = 0
                    loss = 0
                    sum_acc += self.accu
                    sum_acc_iou += self.accu_iou
                    sum_acc_ellip += self.ellip_acc
                    total_loss += loss
                    print('\r' + 12 * ' ', end='')
                    print('epoch %5d\t learning_rate = %g\t step = %4d\t loss = %.4f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%\t valid_ellip_acc = %.2f' % (epoch, self.learning_rate, step, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), (sum_acc_ellip/count)))
        
            #End valid data
            #count -= 1
            print('epoch %5d\t learning_rate = %g\t loss = %.4f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%\t valid_ellip_acc = %.2f' % 
            (epoch, self.learning_rate, total_loss/count, sum_acc/count, sum_acc_iou/count, sum_acc_ellip/count))
            print('Take time %3.1f' % (time.time() - t0))


        except tf.errors.OutOfRangeError:
            print('Error!')
        
        
    
    def train_one_epoch(self, sess, writer, epoch, step):
        print('sub_train_one_epoch')
        sum_acc = 0
        sum_acc_iou = 0
        sum_acc_label = 0
        sum_acc_ellip = 0
        count = 0
        total_loss = 0
        t0 =time.time()
        mean_acc = 0
        mean_acc_iou = 0
        mean_acc_label = 0
        mean_acc_ellip = 0

        if_epoch = cfgs.test_accu
        #if epoch % 5 == 0:
        #    if_epoch = True

        try:
            while count<self.per_e_train_batch:
                step += 1
                count += 1
                
                #1. train
                images_, cur_ims_, ellip_infos_, annos_, labels, filenames = sess.run([self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations, self.train_labels, self.train_filenames])
                
                #cv2.imwrite('%s_anno.bmp' % filenames[0], annos_[0]*255)
                #pdb.set_trace()
                cur_batch_size = images_.shape[0]
                if cur_batch_size == cfgs.batch_size:
                    coord_map_x_cur, coord_map_y_cur = self.coord_map_x, self.coord_map_y
                else:
                    coord_map_x_cur, coord_map_y_cur = self.generate_coord_map(cur_batch_size)

                if cur_batch_size != cfgs.batch_size:
                    #pdb.set_trace()
                    last_idx = cur_batch_size - 1
                    last_im = np.expand_dims(images_[last_idx], axis=0)
                    last_cur_im = np.expand_dims(cur_ims_[last_idx], axis=0)
                    last_ellip_info = np.expand_dims(ellip_infos_[last_idx], axis=0)
                    last_anno = np.expand_dims(annos_[last_idx], axis=0)
                    last_label = np.expand_dims(labels[last_idx], axis=0)
                    last_fn = np.expand_dims(filenames[last_idx], axis=0)
                    for i in range(cur_batch_size, cfgs.batch_size):
                        images_ = np.append(images_, last_im, axis=0)
                        cur_ims_ = np.append(cur_ims_, last_cur_im, axis=0)
                        ellip_infos_ = np.append(ellip_infos_, last_ellip_info, axis=0)
                        annos_ = np.append(annos_, last_anno, axis=0)
                        labels = np.append(labels, last_label, axis=0)
                        filenames = np.append(filenames, last_fn, axis=0)


                    
                ellip_info_low = np.min(ellip_infos_[:, 2:], 1) / 2
                ellip_info_high = np.max(ellip_infos_[:, 2:], 1) / 2
                ellip_info_mean = np.mean(ellip_infos_[:, 2:], 1) / 4
                
 
                #generate mask ims
                mask_ims, pred_anno_, pred_seq_pro_, summary_str = sess.run([self.mask_ims, self.mask_anno, self.pro, self.summary_op],
 

                                                                feed_dict={self.images: images_, 
                                                                         self.annotations: annos_, self.lr: self.learning_rate,
                                                                         
                                                                         self.class_labels: labels,
                                                                         self.keep_prob: cfgs.keep_prob,
                                                                         self.input_keep_prob: 1,
                                                                         self.cur_batch_size: cfgs.batch_size,
                                                                         self.coord_x_tensor: coord_map_x_cur,
                                                                         self.coord_y_tensor: coord_map_y_cur,

                                                                         #self.ellip_low: ellip_info_low,
                                                                         #self.ellip_high: ellip_info_high,
                                                                         self.ellip_axis: ellip_info_mean})
                

                #print('laebls: ', labels)
                #print('pred_labels: ', self.pred_label_)
                #self.im_mask_view(filenames, pred_anno_, pred_seq_pro_, images_, step)
                #cv2.imwrite('%s_anno.bmp' % filenames[0], pred_anno_[0]*127)
                #pdb.set_trace()

                #2. calculate accurary
                #self.calculate_acc(mask_ims.copy(), filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_, if_epoch=if_epoch)
                #self.calculate_acc(annos_, filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_, if_epoch=if_epoch)
                self.calculate_acc(cur_ims_[:, 2:cfgs.ANNO_IMAGE_SIZE[0]+2, :, :], filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_, if_epoch=if_epoch)


 

                self.accu = 0
                self.accu_iou = 0
                loss = 0
                self.acc_label_ = 0

                sum_acc += self.accu
                sum_acc_iou += self.accu_iou
                sum_acc_label += self.acc_label_
                sum_acc_ellip += self.ellip_acc
                mean_acc = sum_acc/count
                mean_acc_iou = sum_acc_iou/count
                mean_acc_label = sum_acc_label/count
                mean_acc_ellip = sum_acc_ellip/count
                #3. calculate loss
                total_loss += loss
                
                #4. time consume
                time_consumed = time.time() - t0
                time_per_batch = time_consumed/count

                #5. check if change learning rate
                if count % 100 == 0:
                    self.try_update_lr()
                #6. summary
                writer.add_summary(summary_str, global_step=step)

                #6. print
                #print('\r' + 2 * ' ', end='')
                line = 'Train epoch %2d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t m_loss=%.4f\t acc = %.2f%%\t iou_acc = %.2f%%\t acc_label = %.2f%%\t ellip_acc = %.2f\t time = %.2f' % (epoch, self.learning_rate, step, count, loss, (total_loss/count), mean_acc, mean_acc_iou, mean_acc_label, mean_acc_ellip, time_per_batch)
                utils.clear_line(len(line))
                print('\r' + line, end='')
                #print('epoch %5d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t mean_loss=%.4f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_ellip_acc = %.2f\t time = %.2f' % (epoch, self.learning_rate, step, count, loss, (total_loss/count), mean_acc, mean_acc_iou, mean_acc_ellip, time_per_batch))
            
            #End one epoch
            #count -= 1
            print('\nepoch %5d\t learning_rate = %g\t mean_loss = %.4f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_acc_label = %.2f%%\t train_ellip_acc = %.2f' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), (sum_acc_label/count), (sum_acc_ellip/count)))
            print('Take time %3.1f' % (time.time() - t0))

        except tf.errors.OutOfRangeError:
            print('Error!')
            count -= 1
            print('epoch %5d\t learning_rate = %g\t mean_loss = %.3f\t train_accuracy = %.2f%%\t train_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))
            print('Take time %3.1f' % (time.time() - t0))
     
        return step


    #------------------------------------------------------------------------------

#Main function
def main():
 
    with tf.device('/gpu:0'):
        #train_records, valid_records = scene_parsing_fg.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)

        print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build()
        model.train()

def vis_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of valid records is %d.' %  len(valid_records))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build_vis()
        model.vis()

def haus_main():
 
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build_h_loss()
        model.train()

def vis_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of valid records is %d.' %  len(valid_records))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build_vis()
        model.vis()



def video_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_video_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of video records is %d.' %  len(valid_records))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build_video()
        model.vis_video()

def generate_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build()
        model.generate()



if __name__ == '__main__':
    if cfgs.mode == 'train':
        main()
    elif cfgs.mode == 'visualize':
        vis_main()
    elif cfgs.mode == 'vis_video':
        video_main()
    elif cfgs.mode == 'generate_re':
        generate_main()
    elif cfgs.mode == 'haus_loss':
        haus_main()

