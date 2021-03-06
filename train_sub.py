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
#import BatchReader_multi as dataset
from BatchReader_multi_ellip import *
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

from train_parent import FCNNet

try:
    from .cfgs.config_train_m import cfgs
except Exception:
    from cfgs.config_train_m import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class ImFCNNet(FCNNet):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir):

        FCNNet.__init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir)

        #mask
        self.inference_name = 'inference_name'
        self.channel = 3
        self.images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, 3], name='input_image')
        
    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_ellip_infos, self.train_annotations, self.train_filenames = get_data_cache(self.train_records, self.batch_size, False, 'get_data_train_mask')
            self.valid_images, self.valid_ellip_infos, self.valid_annotations, self.valid_filenames = get_data_cache(self.valid_records, self.batch_size, False, 'get_data_valid_mask')

    def get_data_vis(self):
        with tf.device('/cpu:0'):
            self.vis_images, self.vis_ellip_infos, self.vis_annotations, self.vis_filenames, self.vis_init = get_data_vis_mask(self.valid_records, self.batch_size)


    def get_data_video(self):
        with tf.device('/cpu:0'):
            self.video_images, self.video_cur_ims, self.video_filenames, self.video_init = get_data_video_mask(self.valid_records, self.batch_size)

    #2. loss 
    def loss(self):
        self.logits = self.inference(self.images, self.inference_name, self.channel, self.keep_prob)
        self.pro = tf.nn.softmax(self.logits)
        self.pred_annotation = tf.expand_dims(tf.argmax(self.pro, dimension=3, name='pred'), dim=3)
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                        labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                        name='entropy_mask')))


    #3. accuracy
    def calculate_acc(self, im, filenames, pred_anno, anno, gt_ellip_info):
        with tf.name_scope('ellip_accu'):
            #self.accu_iou, self.accu = accu.caculate_accurary(pred_anno, anno)
            #self.ellip_acc = accu.caculate_ellip_accu(im, filenames, pred_anno, gt_ellip_info)
            self.ellip_acc =0
            self.accu_iou =0
            self.accu = 0
    #5. build graph
    
    def build(self):
        self.get_data_cache()
        self.loss()
        self.train_optimizer()
        self.summary()

    def build_vis(self):
        self.get_data_vis()
        self.loss_mask()

    def build_video(self):
        self.get_data_video()
        self.loss_mask()
    
    #6. else
    def vis_one_im(self):
        if cfgs.anno:
            im_ = pred_visualize(self.vis_image.copy(), self.vis_pred).astype(np.uint8)
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
        if cfgs.anno_fuse:
            im_ = pred_visualize_choose_color(self.vis_image.copy(), (self.vis_pred+self.vis_pred_cur_max)).astype(np.uint8)
            #im_ = pred_visualize_choose_color(im_, self.vis_pred_cur_max, [255,0,0]).astype(np.uint8)
            utils.save_image(im_, self.re_save_dir_im, name='fuse_' + self.filename + '.jpg')
        if cfgs.fuse_ellip:
            im_ellip = fit_ellipse_findContours(self.vis_image.copy(), np.expand_dims(self.vis_pred+self.vis_pred_cur_max, axis=2).astype(np.uint8))
            utils.save_image(im_ellip, self.re_save_dir_ellip, name='fuse_ellip_' + self.filename + '.jpg')
 

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
                images_, annos_, cur_ims_, filenames_ = sess.run([self.vis_images, self.vis_annotations, self.vis_cur_ims, self.vis_filenames])
                pred_anno, pred_prob = sess.run([self.pred_annotation, self.cur_pro], feed_dict={self.images: cur_ims_, self.mask_images: images_})
                pred_anno = np.squeeze(pred_anno, axis=3)
                
                for i in range(len(pred_anno)):
                    self.filename = filenames_[i].strip().decode('utf-8')
                    self.vis_image = cur_ims_[i]
                    self.vis_anno = annos_[i]
                    self.vis_pred = pred_anno[i]
                    self.vis_pred_prob = pred_prob[i]
                    self.vis_one_im()
        except tf.errors.OutOfRangeError:
            pass

                    
    #Visualize the video result
    def vis_video_once(self, sess):
        sess.run(self.video_init)
        
        self.create_re_dir()

        count = 0
        t0 = time.time()

        try:
            total_loss = 0
            while True:
                count += 1
                images_, filenames_ = sess.run([self.video_images, self.video_cur_ims, self.video_filenames])
                pred_anno, pred_cur_max_anno, pred_prob = sess.run([self.pred_annotation, self.pred_cur_max_anno, self.pro_mask], feed_dict={self.images: cur_ims_, self.mask_images: images_})
                #pred_anno, pred_prob = sess.run([self.pred_cur_anno, self.cur_pro], feed_dict={self.images: cur_ims_, self.mask_images: images_})
                pred_anno = np.squeeze(pred_anno, axis=3)
                pred_cur_max_anno = np.squeeze(pred_cur_max_anno, axis=3)

                for i in range(len(pred_anno)):
                    self.filename = filenames_[i].strip().decode('utf-8')
                    self.vis_image = cur_ims_[i]
                    self.vis_pred = pred_anno[i]
                    self.vis_pred_cur_max = pred_cur_max_anno[i]
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
            while count<self.per_e_valid_batch:
                count +=1
                images_, ellip_infos_, annos_, filenames = sess.run([self.valid_images, self.valid_ellip_infos, self.valid_annotations, self.valid_filenames])
                pred_anno, summary_str, loss= sess.run(
                fetches=[self.pred_annotation, self.summary_op, self.loss],
                feed_dict={self.images: images_, 
                           self.annotations: annos_, self.lr: self.learning_rate})


                writer.add_summary(summary_str, global_step=step)
                self.calculate_acc(images_.copy(), filenames, pred_anno, annos_, ellip_infos_)
                sum_acc += self.accu
                sum_acc_iou += self.accu_iou
                sum_acc_ellip += self.ellip_acc
                total_loss += loss
                print('\r' + 12 * ' ', end='')
                print('epoch %5d\t learning_rate = %g\t step = %4d\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%\t valid_ellip_acc = %.2f%%' % (epoch, self.learning_rate, step, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), (sum_acc_ellip/count)))
        
            #End valid data
            print('epoch %5d\t learning_rate = %g\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%' % 
            (epoch, self.learning_rate, total_loss/count, sum_acc/count, sum_acc_iou/count))
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
            while count<self.per_e_train_batch:
                step += 1
                count += 1
                #1. train
                images_, ellip_infos_, annos_, filenames = sess.run([self.train_images, self.train_ellip_infos, self.train_annotations, self.train_filenames])
                
                pred_anno, pred_pro, summary_str, loss, _ = sess.run([self.pred_annotation, self.pro, self.summary_op, self.loss, self.train_op],
                                                            feed_dict={self.images: images_, 
                                                                       self.annotations: annos_, self.lr: self.learning_rate})


                #pred_anno = np.squeeze(pred_anno, axis=3)
                if count % 10 == 0:
                    choosen = random.randint(0, self.batch_size-1)
                    fn = filenames[choosen].strip().decode('utf-8')
                    pred_anno_im = (pred_anno[choosen]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('image', str(step)+'_'+fn+'.bmp'), pred_anno_im)
                    heat_map = density_heatmap(pred_pro[choosen, :, :, 1])
                    cv2.imwrite(os.path.join('image', str(step)+'_heatseq_'+fn+'.bmp'), heat_map)

                #2. calculate accurary
                self.calculate_acc(images_.copy(), filenames, pred_anno, annos_, ellip_infos_)
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
            print('epoch %5d\t learning_rate = %g\t mean_loss = %.3f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_ellip_acc = %.2f' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), mean_acc_ellip))
            print('Take time %3.1f' % (time.time() - t0))

        except tf.errors.OutOfRangeError:
            print('Error!')
            
        return step
    
    


#-------------------------------------------------------------------------------

#Main function
def main():
 
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = ImFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build()
        model.train()

def vis_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of valid records is %d.' %  len(valid_records))
        model = ImFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build_vis()
        model.vis()

def video_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of video records is %d.' %  len(valid_records))
        model = ImFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob, cfgs.logs_dir)
        model.build_video()
        model.vis_video()

if __name__ == '__main__':
    if cfgs.mode == 'train':
        main()
    elif cfgs.mode == 'visualize':
        vis_main()
    elif cfgs.mode == 'vis_video':
        video_main()

