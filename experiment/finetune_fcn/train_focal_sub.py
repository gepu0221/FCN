#This code is for training seq+cur=7 together.
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import TensorflowUtils as utils
import read_data as scene_parsing
import datetime
import pdb
from BatchReader_multi_ellip import *
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

#from train_seq_parent import FCNNet
from train_resnet_parent import Res101FCNNet as FCNNet

try:
    from .cfgs.config_train_resnet import cfgs
except Exception:
    from cfgs.config_train_resnet import cfgs

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
        accu.create_ellipse_f()

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

        a_w = (1 - 2*self.at) * tf.cast(tf.squeeze(self.annotations, squeeze_dims=[3]), tf.float32) + self.at
      
        loss_weight = tf.pow(1-tf.reduce_sum(self.pro * tf.one_hot(tf.squeeze(self.annotations, squeeze_dims=[3]), self.NUM_OF_CLASSESS), 3), self.gamma)
     
    
        self.loss = tf.reduce_mean(loss_weight * a_w * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                       labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                       name="entropy"))


        #test 
        #show the lower probility area
        self.pro_lower = tf.add(self.pro , cfgs.offset)
        self.pred_anno_lower = tf.expand_dims(tf.argmax(self.pro_lower, dimension=3, name='pred_lower'), dim=3)
        #sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]]
        #im_comp = tf.ones(sz, dtype=tf.int32)
        #self.pre_anno_lower = tf.expand_dims(tf.where(tf.less_equal(self.logits, 0), 1-im_comp, im_comp), dim=3)
        
    #3. accuracy
    def calculate_acc(self, im, filenames, pred_anno, pred_pro, anno, gt_ellip_info, if_valid=False):
        with tf.name_scope('ellip_accu'):
            self.accu_iou, self.accu = accu.caculate_accurary(pred_anno, anno)
            self.ellip_acc = accu.caculate_ellip_accu(im, filenames, pred_anno, pred_pro, gt_ellip_info, if_valid)
            #self.accu_iou = 0
            #self.accu = 0
            #self.ellip_acc = 0
    
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
    
    def build_h_loss(self):
        self.get_data_cache()
        self.loss()
        self.h_loss()

    #6. else
    def view_one(self, fn, pred_anno, pred_pro, im, step):
        path_ = os.path.join(cfgs.view_path, 'train')
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


    def vis_one_im(self):
        if cfgs.anno:
            im_ = pred_visualize(self.vis_image.copy(), self.vis_pred).astype(np.uint8)
            utils.save_image(im_, self.re_save_dir_im, name='inp_' + self.filename + '.jpg')
        if cfgs.fit_ellip:
            #im_ellip = fit_ellipse_findContours(self.vis_image.copy(), np.expand_dims(self.vis_pred, axis=2).astype(np.uint8))
            im_ellip = fit_ellipse(self.vis_image.copy(), np.expand_dims(self.vis_pred, axis=2).astype(np.uint8))
            
            utils.save_image(im_ellip, self.re_save_dir_ellip, name='ellip_' + self.filename + '.jpg')
        if cfgs.heatmap:
            heat_map = density_heatmap(self.vis_pred_prob[:, :, 1])
            utils.save_image(heat_map, self.re_save_dir_heat, name='heat_' + self.filename + '.jpg')
        if cfgs.trans_heat and cfgs.heatmap:
            trans_heat_map = translucent_heatmap(self.vis_image.astype(np.uint8).copy(), heat_map.astype(np.uint8).copy())
            utils.save_image(trans_heat_map, self.re_save_dir_transheat, name='trans_heat_' + self.filename + '.jpg')
            utils.save_image(im_, self.re_save_dir_im, name='fuse_' + self.filename + '.jpg')
        if cfgs.lower_anno:
            im_ = pred_visualize(self.vis_image.copy(), self.vis_pred_lower).astype(np.uint8)
            utils.save_image(im_, self.re_save_dir_im, name='inp_lower_' + self.filename + '.jpg')
        if cfgs.fit_ellip_lower:
            im_ellip = fit_ellipse_findContours(self.vis_image.copy(), np.expand_dims(self.vis_pred_lower, axis=2).astype(np.uint8))
            utils.save_image(im_ellip, self.re_save_dir_ellip, name='ellip_lower' + self.filename + '.jpg')
        
            

    #Visualize the result
    def visualize(self, sess):
        sess.run(self.vis_init)
        
        self.create_re_dir()

        count = 0
        t0 = time.time()

        try:
            total_loss = 0
            while True:
                count += 1
                images_, cur_ims, annos_, filenames_ = sess.run([self.vis_images, self.vis_cur_images, self.vis_annotations, self.vis_filenames])
                pred_anno, pred_prob = sess.run([self.pred_annotation, self.pro], feed_dict={self.images: images_})
                pred_anno = np.squeeze(pred_anno, axis=3)
                
                for i in range(len(pred_anno)):
                    self.filename = filenames_[i].strip().decode('utf-8')
                    self.vis_image = cur_ims[i]
                    self.vis_anno = annos_[i]
                    self.vis_pred = pred_anno[i]
                    self.vis_pred_prob = pred_prob[i]
                    self.vis_one_im()
        except tf.errors.OutOfRangeError:
            pass

                    

    #Evaluate all validation dataset once 
    def valid_once(self, sess, writer, epoch, step):

        count = 0
        sum_acc = 0
        sum_acc_iou = 0
        sum_acc_ellip = 0
        t0 = time.time()

        try:
            total_loss = 0
            #self.per_e_valid_batch = 2
            while count<self.per_e_valid_batch:
                count +=1
                images_, cur_ims, ellip_infos_, annos_, filenames = sess.run([self.valid_images, self.valid_cur_ims, self.valid_ellip_infos, self.valid_annotations, self.valid_filenames])
                pred_anno, pred_seq_pro, summary_str, loss= sess.run(
                #fetches=[self.pred_annotation, self.pro, self.summary_op, self.loss],
                fetches=[self.pred_anno_lower, self.pro, self.summary_op, self.loss],
                feed_dict={self.images: images_, 
                           self.annotations: annos_, self.lr: self.learning_rate,
                           self.keep_prob: 1,
                           self.input_keep_prob: 1})
                
                #View result
                self.view_valid(filenames, pred_anno, pred_seq_pro, images_, step)

                writer.add_summary(summary_str, global_step=step)
                self.calculate_acc(cur_ims.copy(), filenames, pred_anno, pred_seq_pro, annos_, ellip_infos_, True)
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
        sum_acc_ellip = 0
        count = 0
        total_loss = 0
        t0 =time.time()
        mean_acc = 0
        mean_acc_iou = 0
        mean_acc_ellip = 0
        try:
            #self.per_e_train_batch = 2
            while count<self.per_e_train_batch:
                step += 1
                count += 1
                
                #1. train
                images_, cur_ims_, ellip_infos_, annos_, filenames = sess.run([self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations, self.train_filenames])
                
                #cv2.imwrite('%s_anno.bmp' % filenames[0], annos_[0]*255)
                #pdb.set_trace()
                pred_anno_, pred_seq_pro_, summary_str, loss, _ = sess.run([self.pred_anno_lower, self.pro, self.summary_op, self.loss, self.train_op],
                #pred_anno_, pred_seq_pro_, summary_str, loss = sess.run([self.pred_annotation, self.pro, self.summary_op, self.loss],
                #pred_anno_, pred_seq_pro_, summary_str, loss = sess.run([self.pred_anno_lower, self.pro, self.summary_op, self.loss],
                                                                feed_dict={self.images: images_, 
                                                                         self.annotations: annos_, self.lr: self.learning_rate,
                                                                         self.keep_prob: 1,
                                                                         self.input_keep_prob: 1})

          
                self.view(filenames, pred_anno_, pred_seq_pro_, images_, step)
                #2. calculate accurary
                
                self.calculate_acc(cur_ims_.copy(), filenames, pred_anno_, pred_seq_pro_, annos_, ellip_infos_)
                sum_acc += self.accu
                sum_acc_iou += self.accu_iou
                sum_acc_ellip += self.ellip_acc
                mean_acc = sum_acc/count
                mean_acc_iou = sum_acc_iou/count
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
                print('epoch %5d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t mean_loss=%.4f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_ellip_acc = %.2f\t time = %.2f' % (epoch, self.learning_rate, step, count, loss, (total_loss/count), mean_acc, mean_acc_iou, mean_acc_ellip, time_per_batch))
            
            #End one epoch
            #count -= 1
            print('epoch %5d\t learning_rate = %g\t mean_loss = %.4f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_ellip_acc = %.2f' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), (sum_acc_ellip/count)))
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

