#This is FCN for sequence as mask created on 20180728
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import random
import TensorflowUtils as utils
import read_data as scene_parsing
import datetime
import pdb
#import BatchReader_multi as dataset
from BatchReader_multi import *
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, pred_visualize_choose_color, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

try:
    from .cfgs.config_train_mask import cfgs
except Exception:
    from cfgs.config_train_mask import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class FCNNet(object):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob):
        self.max_epochs = max_epochs
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.NUM_OF_CLASSESS = n_classes
        self.IMAGE_SIZE = im_sz
        self.graph = tf.get_default_graph()
        self.lr= tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.learning_rate = float(init_lr)
        self.mode = mode
        self.logs_dir = cfgs.logs_dir
        self.seq_num = cfgs.seq_num
        self.current_itr_var = tf.Variable(0, dtype=tf.int32, name='current_itr', trainable=True)
        self.cur_epoch = tf.Variable(1, dtype=tf.int32, name='cur_epoch', trainable=False)
        #mask
        self.inference_name = 'inference_name'
        self.channel = 3
        self.seq_infer_name = 'seq_infer_name'
        self.seq_channel = self.seq_num

        self.train_records = train_records
        self.valid_records = valid_records
        self.per_e_train_batch = len(self.train_records)/self.batch_size
        self.per_e_valid_batch = len(self.valid_records)/self.batch_size

        self.images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, 3], name='input_image')
        self.mask_images = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, self.seq_num], name='mask_input_image')
        self.annotations = tf.placeholder(tf.int32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, 1], name='annotations')

        if self.mode == 'visualize' or 'vis_video':
            self.result_dir = cfgs.result_dir
        self.at = cfgs.at
        self.gamma = cfgs.gamma

        
    #1. get data
    def get_data_cache(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_annotations, self.train_filenames = get_data_cache(self.train_records, self.batch_size, False, 'get_data_train')
            self.valid_images, self.valid_annotations, self.valid_filenames = get_data_cache(self.valid_records, self.batch_size, False, 'get_data_valid')

    def get_data_mask(self):
        with tf.device('/cpu:0'):
            self.train_images, self.train_cur_ims, self.train_annotations, self.train_filenames = get_data_mask(self.train_records, self.batch_size, False, 'get_data_train_mask')
            self.valid_images, self.valid_cur_ims, self.valid_annotations, self.valid_filenames = get_data_mask(self.valid_records, self.batch_size, False, 'get_data_valid_mask')

    def get_data(self):
        with tf.device('/cpu:0'):
            self.images, self.annotations, self.filenames, self.train_init, self.valid_init = get_data_(self.train_records, self.valid_records, self.batch_size)
    
    def get_data_vis_mask(self):
        with tf.device('/cpu:0'):
            self.vis_images, self.vis_cur_ims, self.vis_annotations, self.vis_filenames, self.vis_init = get_data_vis_mask(self.valid_records, self.batch_size)

    def get_data_video_mask(self):
        with tf.device('/cpu:0'):
            self.video_images, self.video_cur_ims, self.video_filenames, self.video_init = get_data_video_mask(self.valid_records, self.batch_size)

    #2. net
    def vgg_net(self, weights, image):
        layers = (
            #'conv1_1', 'relu1_1',
            'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i+2][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = utils.avg_pool_2x2(current)
            net[name] = current

        return net


    def inference(self, images, inference_name, channel,  keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up vgg initialized conv layers ...")
        model_data = utils.get_model_data(cfgs.model_dir, MODEL_URL)

        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean)
        self.mean_ = mean_pixel
        weights = np.squeeze(model_data['layers'])

        #processed_image = utils.process_image(self.images, mean_pixel)

        #with tf.variable_scope("inference"):
        with tf.variable_scope(inference_name):
            W1 = utils.weight_variable([3, 3, channel, 64], name="W1")
            b1 = utils.bias_variable([64], name="b1")
            conv1 = utils.conv2d_basic(images, W1, b1)
            #conv1 = utils.conv2d_basic(processed_image, W1, b1)
            relu1 = tf.nn.relu(conv1, name='relu1')
            
            #pretrain
            image_net = self.vgg_net(weights, relu1)
            
            conv_final_layer = image_net["conv5_3"]

            pool5 = utils.max_pool_2x2(conv_final_layer)

            W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
            b6 = utils.bias_variable([4096], name="b6")
            conv6 = utils.conv2d_basic(pool5, W6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
            b7 = utils.bias_variable([4096], name="b7")
            conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            '''
            if cfgs.debug:
                utils.add_activation_summary(relu7)'''
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
            b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
            conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            deconv_shape1 = image_net["pool4"].get_shape()
            W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
            b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
            conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

            deconv_shape2 = image_net["pool3"].get_shape()
            W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
            conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

            shape = tf.shape(images)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
            W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
            b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
            conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        #self.pred_annotation = tf.expand_dims(annotation_pred, dim=3)
        #self.logits = conv_t3
        return conv_t3

    def mask_mul(self):
        self.logits = self.inference(self.images, self.inference_name, self.channel, self.keep_prob)
        self.seq_logits = self.inference(self.mask_images, self.seq_infer_name, self.seq_channel, self.keep_prob)
        self.cur_pro = tf.nn.softmax(self.logits)
        self.cur_pro_max = tf.subtract(self.cur_pro, cfgs.offset)
        self.seq_pro = tf.nn.softmax(self.seq_logits)

        self.pred_cur_anno = tf.expand_dims(tf.argmax(self.cur_pro, dimension=3, name='pred_anno_cur'), dim=3)
        self.pred_cur_max_anno = tf.expand_dims(tf.argmax(self.cur_pro_max, dimension=3, name='pred_anno_cur_max'), dim=3)
        self.pred_seq_anno = tf.expand_dims(tf.argmax(self.seq_pro, dimension=3, name='pred_anno_seq'), dim=3)

        self.logits_mask = tf.multiply(self.logits, self.seq_pro, name='logits_mask')
        
    
    #3. optmizer
    def train_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        var_list = tf.trainable_variables()
        grads = optimizer.compute_gradients(self.loss_seq_mask, var_list=var_list)
        #pdb.set_trace()
        self.train_op = optimizer.apply_gradients(grads)

    
    #4. loss
    #loss for mask
    def loss_mask(self):
        self.pro_mask = tf.nn.softmax(self.logits_mask)
        self.pred_annotation = tf.expand_dims(tf.argmax(self.pro_mask, dimension=3, name='pred_mask'), dim=3)
        self.loss_mask = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_mask,
                                                                                        labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                        name='entropy_mask')))
    #only for seq
    def loss_seq_mask(self):
        self.seq_logits = self.inference(self.mask_images, self.seq_infer_name, self.seq_channel, self.keep_prob)
        self.loss_seq_mask = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.seq_logits_mask,
                                                                                        labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                                        name='entropy_mask')))
 
    #5. evaluation
    def calculate_acc(self, pred_anno, anno):
        with tf.name_scope('accu'):
            self.accu_iou, self.accu = accu.caculate_accurary(pred_anno, anno)
    
    #not use
    def eval(self):
        with tf.name_scope('eval'):
            is_correct = tf.equal(tf.cast(self.pred_annotation, tf.int32), self.annotations)
            sum_ = tf.cast(tf.reduce_sum(tf.cast(is_correct, tf.int32)), tf.float32)
            self.acc = tf.multiply(sum_, 100/(self.IMAGE_SIZE*self.IMAGE_SIZE*self.batch_size))

    #7. summary
    def summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('train_loss_mask', self.loss_mask)
            #tf.summary.scalar('accu', self.accu)
            #tf.summary.scalar('iou_accu', self.accu_iou)
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.summary_op = tf.summary.merge_all()
    
    #8. graph build
    #test mask
    def build_mask(self):
        #build the graph
        self.get_data_mask()
        self.mask_mul()
        self.loss_mask()
        self.train_optimizer()
        self.summary()

    def build_mask_seq(self):
        self.get_data_mask()
        self.loss_seq_max()
        self.train_optimizer()
        
        

    def build_vis(self):
        #bulid the visualize graph
        self.get_data_vis_mask()
        self.mask_mul()
        self.loss_mask()
        self.train_optimizer()
        self.summary()
        

    def build_video(self):
        #build the video graph
        self.get_data_video_mask()
        self.mask_mul()
        self.loss_mask()
        
        
    
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
                images_, cur_ims_, filenames_ = sess.run([self.video_images, self.video_cur_ims, self.video_filenames])
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
        t0 = time.time()

        try:
            total_loss = 0
            while count<self.per_e_valid_batch:
                count +=1
                images_, cur_ims, annos_, filenames = sess.run([self.valid_images, self.valid_cur_ims, self.valid_annotations, self.valid_filenames])
                pred_anno, summary_str, loss= sess.run(
                fetches=[self.pred_annotation, self.summary_op, self.loss_mask],
                feed_dict={self.images: cur_ims, self.mask_images: images_, 
                           self.annotations: annos_, self.lr: self.learning_rate})


                writer.add_summary(summary_str, global_step=step)
                self.calculate_acc(pred_anno, annos_)
                sum_acc += self.accu
                sum_acc_iou += self.accu_iou
                total_loss += loss
                print('\r' + 32 * ' ', end='')
                print('epoch %5d\t learning_rate = %g\t step = %4d\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, step, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))
        
            #End valid data
            count -= 1
            print('epoch %5d\t learning_rate = %g\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%' % 
            (epoch, self.learning_rate, total_loss/count, sum_acc/count, sum_acc_iou/count))
            print('Take time %3.1f' % (time.time() - t0))


        except tf.errors.OutOfRangeError:
            print('Error!')
            count -= 1
            print('epoch %5d\t learning_rate = %g\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%' % 
            (epoch, self.learning_rate, total_loss/count, sum_acc/count, sum_acc_iou/count))
            print('Take time %3.1f' % (time.time() - t0))
    

    
    
    def train_one_epoch(self, sess, writer, epoch, step):
        sum_acc = 0
        sum_acc_iou = 0
        count = 0
        total_loss = 0
        t0 =time.time()
        mean_acc = 0
        mean_acc_iou = 0
        try:
            while count<self.per_e_train_batch:
                step += 1
                count += 1
                #1. train
                images_, cur_ims, annos_, filenames = sess.run([self.train_images, self.train_cur_ims, self.train_annotations, self.train_filenames])
                
                pred_anno, pred_cur_anno, pred_seq_pro, summary_str, loss, _ = sess.run([self.pred_annotation, self.pred_cur_anno, self.seq_pro, self.summary_op, self.loss_mask, self.train_op],
                                                            feed_dict={self.images: cur_ims, self.mask_images: images_, 
                                                                       self.annotations: annos_, self.lr: self.learning_rate})


                print(pred_seq_pro[0].shape)
                if count % 10 == 0:
                    choosen = random.randint(0, self.batch_size-1)
                    fn = filenames[choosen].strip().decode('utf-8')
                    #img = images_[0,:,:,0:3].astype(np.uint8)
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    #cv2.imwrite(os.path.join('image', 'im' + str(step) + '.bmp'), img)
                    #anno = annos_[0].astype(np.uint8)
                    #cv2.imwrite(os.path.join('image', 'an' + str(step) + '.bmp'), anno)
                    pred_anno_im = (pred_anno[choosen]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('image', 'mask', str(step)+'_'+fn+'.bmp'), pred_anno_im)
                    pred_cur_im = (pred_cur_anno[choosen]*255).astype(np.uint8)
                    cv2.imwrite(os.path.join('image', 'mask', str(step)+'_cur_'+fn+'.bmp'), pred_cur_im)
                    #print(pred_seq_pro[choosen, :, :, 1])
                    heat_map = density_heatmap(pred_seq_pro[choosen, :, :, 1])
                    cv2.imwrite(os.path.join('image', 'mask', str(step)+'_heatseq_'+fn+'.bmp'), heat_map)
                    img_seq = images_[choosen]
                    cv2.imwrite(os.path.join('image', 'mask', str(step)+'_seq_'+fn+'.bmp'), img_seq)
                    #pred_seq_im = (pred_seq_anno[choosen]*255).astype(np.uint8)
                    #cv2.imwrite(os.path.join('image', 'mask', str(step)+'_seq_'+fn+'.bmp'), pred_seq_im)

                #2. calculate accurary
                #if count % 10 ==0:
                self.calculate_acc(pred_anno, annos_)
                sum_acc += self.accu
                sum_acc_iou += self.accu_iou
                mean_acc = sum_acc/count
                mean_acc_iou = sum_acc_iou/count
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
                print('\r' + 8 * ' ', end='')
                print('epoch %5d\t lr = %g\t step = %4d\t count = %4d\t loss = %.3f\t mean_loss=%.3f\t train_accuracy = %.2f%%\t train_iou_accuracy = %.2f%%\t time = %.2f' % (epoch, self.learning_rate, step, count, loss, (total_loss/count), mean_acc, mean_acc_iou, time_per_batch))
            
            #End one epoch
            count -= 1
            print('epoch %5d\t learning_rate = %g\t mean_loss = %.3f\t train_accuracy = %.2f%%\t train_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))
            print('Take time %3.1f' % (time.time() - t0))

        except tf.errors.OutOfRangeError:
            print('Error!')
            count -= 1
            print('epoch %5d\t learning_rate = %g\t mean_loss = %.3f\t train_accuracy = %.2f%%\t train_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))
            print('Take time %3.1f' % (time.time() - t0))
     
        return step

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
                    step = self.train_one_epoch(sess, writer, epoch, step)

                    #3.3 save model
                    self.valid_once(sess, writer, epoch, step)
                    self.cur_epoch.load(epoch, sess)
                    self.current_itr_var.load(step, sess)
                    saver.save(sess, self.logs_dir + 'model.ckpt', step)

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

        

#-------------------------------------------------------------------------------

#Main function
def main():
 
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = FCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob)
        model.build_mask()
        model.train()

def vis_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of valid records is %d.' %  len(valid_records))
        model = FCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob)
        model.build_vis()
        model.vis()

def video_main():
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        print('The number of video records is %d.' %  len(valid_records))
        model = FCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob)
        model.build_video()
        model.vis_video()

if __name__ == '__main__':
    if cfgs.mode == 'train':
        main()
    elif cfgs.mode == 'visualize':
        vis_main()
    elif cfgs.mode == 'vis_video':
        video_main()

