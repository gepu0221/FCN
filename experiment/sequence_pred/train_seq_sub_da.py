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
from BatchReader_multi_da import *
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

#from train_seq_parent import FCNNet
from train_seq_resnet_parent import Res101FCNNet as FCNNet

try:
    from .cfgs.config_train_resnet_da import cfgs
except Exception:
    from cfgs.config_train_resnet_da import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class SeqFCNNet(FCNNet):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir):

        FCNNet.__init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob, logs_dir)

        #mask
        self.seq_num = cfgs.seq_num
        self.channel = 3+self.seq_num
        self.inference_name = 'inference'
        self.images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, cfgs.seq_num+3], name='input_image')

        #mkdir ellipse error path
        acc.create_ellipse_f()
        
    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations, self.train_filenames = get_data_cache(self.train_records, self.batch_size, False, 'get_data_train')
            self.train_da_images, self.train_da_cur_ims, self.train_da_ellip_infos, self.train_da_annotations, self.train_da_filenames = get_data_cache_da(self.train_records, self.batch_size, False, 'get_data_train_da')
            
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

    #3. accuracy
    def calculate_acc(self, im, filenames, pred_anno, anno, gt_ellip_info, if_valid=False):
        with tf.name_scope('ellip_accu'):
            self.accu_iou, self.accu = accu.caculate_accurary(pred_anno, anno)
            self.ellip_acc = accu.caculate_ellip_accu(im, filenames, pred_anno, gt_ellip_info, if_valid)
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
    
    #6. else
    def vis_one_im(self):
        if cfgs.anno:
            im_ = pred_visualize(self.vis_image.copy(), self.vis_pred).astype(np.uint8)
            utils.save_image(im_, self.re_save_dir_im, name='inp_' + self.filename + '.jpg')
        if cfgs.fit_ellip:
            #im_ellip = fit_ellipse_findContours(self.vis_image.copy(), np.expand_dims(self.vis_pred, axis=2).astype(np.uint8))
            im_ellip = fit_ellipse_findContours(self.vis_image.copy(), np.expand_dims(self.vis_pred, axis=2).astype(np.uint8))
            
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
                images_, cur_ims_, filenames_ = sess.run([self.video_images, self.video_cur_ims, self.video_filenames])
                pred_anno, pred_prob, pred_anno_lower = sess.run([self.pred_annotation, self.pro, self.pred_anno_lower], feed_dict={self.images: images_})
                pred_anno = np.squeeze(pred_anno, axis=3)
                pred_anno_lower = np.squeeze(pred_anno_lower)

                for i in range(len(pred_anno)):
                    self.filename = filenames_[i].strip().decode('utf-8')
                    self.vis_image = cur_ims_[i]
                    self.vis_pred = pred_anno[i]
                    self.vis_pred_prob = pred_prob[i]
                    self.vis_pred_lower = pred_anno_lower[i]

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
                           self.annotations: annos_, self.lr: self.learning_rate})

                #pred_anno = np.squeeze(pred_anno, axis=3)
                '''
                for i in range(self.batch_size):
                    fn = filenames[i].strip().decode('utf-8')
                    pred_anno_im = (pred_anno[i]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('ellip_error', 'res_cur_seq_nok_noda_valid', fn+'_anno'+'.bmp'), pred_anno_im)'''
                ''' 
                if count % 10 == 0:
                    choosen = random.randint(0, self.batch_size-1)
                    fn = filenames[choosen].strip().decode('utf-8')
                    pred_anno_im = (pred_anno[choosen]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_val_'+fn+'.bmp'), pred_anno_im)
                    heat_map = density_heatmap(pred_seq_pro[choosen, :, :, 1])
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_val_heatseq_'+fn+'.bmp'), heat_map)

                    im = images_[choosen,:,:,0]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'val_im_'+fn+'.bmp'), im)
                    #generate sequence 
                    im = images_[choosen,:,:,3]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'val_im_seq0'+fn+'.bmp'), im)
                    im = images_[choosen,:,:,4]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'val_im_seq1'+fn+'.bmp'), im)
                    im = images_[choosen,:,:,5]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'val_im_seq2'+fn+'.bmp'), im)
                    im = images_[choosen,:,:,6]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'val_im_seq3'+fn+'.bmp'), im)
                '''



                writer.add_summary(summary_str, global_step=step)
                self.calculate_acc(cur_ims.copy(), filenames, pred_anno, annos_, ellip_infos_, True)
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
                '''
                images_da, cur_ims_da, annos_da, filenames = sess.run([self.train_da_images, self.train_da_cur_ims, self.train_da_annotations, self.train_da_filenames])
                
                pred_anno_da, pred_seq_pro_da, loss_da, _ = sess.run([self.pred_annotation, self.pro, self.loss, self.train_op],
                                                            feed_dict={self.images: images_da, 
                                                                       self.annotations: annos_da, self.lr: self.learning_rate})

                '''
                images_, cur_ims, ellip_infos_, annos_, filenames = sess.run([self.train_images, self.train_cur_ims, self.train_ellip_infos, self.train_annotations, self.train_filenames])
                

                #pred_anno, pred_seq_pro, summary_str, loss = sess.run([self.pred_annotation, self.pro, self.summary_op, self.loss],
                pred_anno, pred_seq_pro, summary_str, loss = sess.run([self.pred_anno_lower, self.pro, self.summary_op, self.loss],
                                                            feed_dict={self.images: images_, 
                                                                       self.annotations: annos_, self.lr: self.learning_rate})

                '''
                if count % 10 == 0:
                    choosen = random.randint(0, self.batch_size-1)
                    fn = filenames[choosen].strip().decode('utf-8')
                    pred_anno_im = (pred_anno_da[choosen]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_'+fn+'.bmp'), pred_anno_im)
                    heat_map = density_heatmap(pred_seq_pro_da[choosen, :, :, 1])
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_heatseq_'+fn+'.bmp'), heat_map)
                    im = images_da[choosen,:,:,0]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_im_'+fn+'.bmp'), im)
                    #generate sequence 
                    im = images_da[choosen,:,:,3]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_im_seq0'+fn+'.bmp'), im)
                    im = images_da[choosen,:,:,4]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_im_seq1'+fn+'.bmp'), im)
                    im = images_da[choosen,:,:,5]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_im_seq2'+fn+'.bmp'), im)
                    im = images_da[choosen,:,:,6]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_im_seq3'+fn+'.bmp'), im)


                    anno_im = (annos_da[choosen]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_anno_'+fn+'.bmp'), anno_im)
                
                for i in range(self.batch_size):
                    fn = filenames[i].strip().decode('utf-8')
                    im = images_da[i,:,:,0]
                    cv2.imwrite(os.path.join('image', 'res_seq_cur', str(step)+'_im_'+fn+'.bmp'), im)'''
                '''       
                for i in range(self.batch_size):
                    fn = filenames[i].strip().decode('utf-8')
                    pred_anno_im = (pred_anno[i]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('ellip_error', 'res_cur_seq_nok_noda', fn+'_anno'+'.bmp'), pred_anno_im)
                 ''' 
                #2. calculate accurary
                self.calculate_acc(cur_ims.copy(), filenames, pred_anno, annos_, ellip_infos_)
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

    def generate_train_one_epoch(self, sess):
        try:
            count = 0
            while count<self.per_e_train_batch:
                count += 1
                images_, cur_ims, filenames = sess.run([self.train_images, self.train_cur_ims, self.train_filenames])

                pred_anno = sess.run(self.pred_annotation,
                                      feed_dict={self.images: images_})
                len_ = len(images_)
                for i in range(len_):
                    fn = filenames[i].strip().decode('utf-8')
                    pred_anno_im = pred_anno[i]
                    pred_anno_im = (np.concatenate((pred_anno_im, pred_anno_im, pred_anno_im), axis=2)).astype(np.uint8)
                    #print(pred_anno_im.shape)
                    cv2.imwrite(os.path.join(cfgs.re_path, fn+'.bmp'), pred_anno_im)

        except tf.errors.OutOfRangeError:
            print('Error!')
     
    def generate_valid_one_epoch(self, sess):
        try:
            count = 0
            while count<self.per_e_valid_batch:
                count += 1
                images_, cur_ims, filenames = sess.run([self.valid_images, self.valid_cur_ims, self.valid_filenames])

                pred_anno = sess.run(self.pred_annotation,
                                      feed_dict={self.images: images_})
                len_ = len(images_)
                for i in range(len_):
                    fn = filenames[i].strip().decode('utf-8')
                    pred_anno_im = pred_anno[i]
                    pred_anno_im = (np.concatenate((pred_anno_im, pred_anno_im, pred_anno_im), axis=2)).astype(np.uint8)
                    cv2.imwrite(os.path.join(cfgs.re_path, fn+'.bmp'), pred_anno_im)


        except tf.errors.OutOfRangeError:
            print('Error!')
    
    def generate(self):
        if not os.path.exists(self.logs_dir):
            print("The logs path '%s' is not found" % self.logs_dir)
            print("Create now..")
            os.makedirs(self.logs_dir)
            print("%s is created successfully!" % self.logs_dir)
        if not os.path.exists(cfgs.re_path):
            print("The result recover path %s is not found" % cfgs.re_path)
            print("Create now..")
            os.makedirs(cfgs.re_path)
            print("%s is created successfilly!" % cfgs.re_path)

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

def video_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
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

