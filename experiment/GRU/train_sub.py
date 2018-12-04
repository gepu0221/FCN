#This code is for training, adding ellipse center to the loss function.
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import models.TensorflowUtils as utils
import datetime
import pdb
from six.moves import xrange
from tools.label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from tools.generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

#from train_seq_parent import FCNNet
from train_GRU_parent import U_Net as FCNNet

try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class SeqFCNNet(FCNNet):

    def __init__(self, mode, max_epochs, batch_size, n_classes, im_sz, init_lr, keep_prob):

        FCNNet.__init__(self, mode, max_epochs, batch_size, n_classes,  im_sz, init_lr, keep_prob)
        
        self.create_view_path()
        
        #accu.create_ellipse_f()
        #self.e_acc = accu.Ellip_acc()

    #1. get data

    #2. loss 
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
        self.accuracy()
        #self.loss()
        #self.acc_label2()
        #self.acc_label2_lower()
        #self.multi_focal_loss()
        #self.train_optimizer()
        #self.summary()

        
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
            filename = fn
        else:
            filename = str(step)+'_'+fn

        pred_anno_im = (pred_anno*127).astype(np.uint8)
        cv2.imwrite(os.path.join(path_, filename+'_anno.bmp'), pred_anno_im)
        cv2.imwrite(os.path.join(path_, filename+'_im.bmp'), im[:,:,0])
        heatmap1 = density_heatmap(pred_pro[:,:,1])
        cv2.imwrite(os.path.join(path_, filename+'_heat1.bmp'), heatmap1)
        heatmap2 = density_heatmap(pred_pro[:,:,2])
        cv2.imwrite(os.path.join(path_, filename+'_heat2.bmp'), heatmap2)

        if cfgs.view_seq:
            for i in range(cfgs.seq_num):
                im_ = im[:,:,self.cur_channel-1+i]
                cv2.imwrite(os.path.join(path_, filename+'seq_'+str(i+1)+'.bmp'), im_)
            
    def view(self, fns, pred_annos, pred_pros, ims, step):
        num_ = fns.shape[0]
        #pdb.set_trace()
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
        pred_anno_im = (pred_anno*127).astype(np.uint8)
        cv2.imwrite(os.path.join(path_, filename+'_anno.bmp'), pred_anno_im)
        cv2.imwrite(os.path.join(path_, filename+'_im.bmp'), im[:,:,0])
        heatmap1 = density_heatmap(pred_pro[:,:,1])
        cv2.imwrite(os.path.join(path_, filename+'_heat1.bmp'), heatmap1)
        heatmap2 = density_heatmap(pred_pro[:,:,2])
        cv2.imwrite(os.path.join(path_, filename+'_heat2.bmp'), heatmap2)

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

    
                        

        
    def train_one_epoch(self, sess, data_loader, data_num, epoch, step):
        
        '''
        Generate once training or validation.
        Args:
            data_loader: training or validation data_loader.
            data_num: number of data.
        '''
        sum_acc = 0
        sum_acc_iou = 0
        sum_acc_ellip = 0
        count = 0
        total_loss = 0
        t0 =time.time()
        mean_acc = 0
        mean_acc_iou = 0
        mean_acc_label = 0
        mean_acc_ellip = 0


        for count in range(1, data_num):
            step += 1
            
            images, images_pad, gt, fn = data_loader.get_next_sequence()

            # Optical flow
            optflow = []
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images[frame], images[frame-1]
                flow = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
                
                optflow.append(flow)

            # Static segmentation(logits)
            static_segm = []
            for frame in range(cfgs.seq_frames):
                im = images_pad[frame]
                
                cv2.imwrite('im.bmp', im[0].astype(np.uint8))
                x_logits, x, x_pro= sess.run([self.static_output, self.static_anno_pred, self.unet_pro],
                                    feed_dict={self.unet_images: im,
                                               self.class_labels: np.array([0])})

                cv2.imwrite('seg.bmp', x*127)
                #pdb.set_trace()

                static_segm.append(x_logits)
            
            # GRFP
            
            rnn_input = {
                self.gru_lr: cfgs.grfp_lr,
                self.gru_input_images_tensor: np.stack(images),
                self.gru_input_flow_tensor: np.stack(optflow),
                self.gru_input_seg_tensor: np.stack(static_segm),
                self.gru_targets: gt
            }
            
            _, loss, pred, pred_pro,\
            self.accu, self.accu_iou \
            = sess.run([self.gru_opt, self.gru_loss, self.gru_pred, self.gru_pred_pro,
                        self.accu_tensor, self.accu_iou_tensor], 
                        feed_dict=rnn_input)

            
            #self.view(np.expand_dims(fn, 0), pred, pred_pro, images[cfgs.seq_frames-1], step)
            x = np.expand_dims(np.expand_dims(x, 0), 3)
            self.view(np.expand_dims(fn, 0), x, x_pro, images[cfgs.seq_frames-1], step)
            
            #2. calculate accurary
            self.ellip_acc = 0
            #self.acc, self.acc_iou = 0
            #self.calculate_acc(images[cfgs.seq_frames, 2:cfgs.ANNO_IMAGE_SIZE[0]+2, :, :].copy(), fn, pred, pred_pro, gt, ellip_infos_, if_epoch=if_epoch)
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

            #5. print
            line = 'Train epoch %2d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t m_loss=%.4f\t acc = %.2f%%\t iou_acc = %.2f%%\t ellip_acc = %.2f\t time = %.2f' % (epoch, self.learning_rate, step, count, loss, (total_loss/count), mean_acc, mean_acc_iou, mean_acc_ellip, time_per_batch)
            utils.clear_line(len(line))
            print('\r' + line, end='')

        #End one epoch
        #count -= 1
        print('\nepoch %5d\t learning_rate = %g\t mean_loss = %.4f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_ellip_acc = %.2f' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), (sum_acc_ellip/count)))
        print('Take time %3.1f' % (time.time() - t0))


        return step


    def valid_once(self, sess, data_loader, data_num, epoch, step):
        
        '''
        Generate once training or validation.
        Args:
            data_loader: training or validation data_loader.
            data_num: number of data.
        '''
        sum_acc = 0
        sum_acc_iou = 0
        sum_acc_ellip = 0
        count = 0
        total_loss = 0
        t0 =time.time()
        mean_acc = 0
        mean_acc_iou = 0
        mean_acc_label = 0
        mean_acc_ellip = 0


        for count in range(1, data_num):
            step += 1
            
            images, images_pad, gt, fn = data_loader.get_next_sequence()

            # Optical flow
            optflow = []
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images[frame], images[frame-1]
                flow = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
                optflow.append(flow)

            # Static segmentation(logits)
            static_segm = []
            for frame in range(cfgs.seq_frames):
                im = images_pad[frame]
                x_logits = sess.run(self.static_output,
                                    feed_dict={self.unet_images: im})

                static_segm.append(x_logits)
            
            # GRFP
            rnn_input = {
                self.gru_lr: cfgs.grfp_lr,
                self.gru_input_images_tensor: np.stack(images),
                self.gru_input_flow_tensor: np.stack(optflow),
                self.gru_input_seg_tensor: np.stack(static_segm),
                self.gru_targets: gt
            }
            
            loss, pred, pred_pro,\
            self.accu, self.accu_iou \
            = sess.run([self.gru_loss, self.gru_pred, self.gru_pred_pro,
                        self.accu_tensor, self.accu_iou_tensor], 
                        feed_dict=rnn_input)


            #self.view(fn, pred, pred_pro, images[cfgs.seq_frames], step)

            #2. calculate accurary
            self.ellip_acc = 0
            #self.calculate_acc(images[cfgs.seq_frames, 2:cfgs.ANNO_IMAGE_SIZE[0]+2, :, :].copy(), fn, pred, pred_pro, gt, ellip_infos_, if_epoch=if_epoch)
            sum_acc += self.accu
            sum_acc_iou += self.accu_iou
            sum_acc_ellip += self.ellip_acc
            mean_acc = sum_acc/count
            mean_acc_iou = sum_acc_iou/count
            mean_acc_ellip = sum_acc_ellip/count
            #3. calculate loss
            total_loss += loss
            pdb.set_trace()   
            #4. time consume
            time_consumed = time.time() - t0
            time_per_batch = time_consumed/count

            #5. print
            line = 'Valid epoch %2d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t m_loss=%.4f\t acc = %.2f%%\t iou_acc = %.2f%%\t ellip_acc = %.2f\t time = %.2f' % (epoch, self.learning_rate, step, count, loss, (total_loss/count), mean_acc, mean_acc_iou, mean_acc_ellip, time_per_batch)
            utils.clear_line(len(line))
            print('\r' + line, end='')

        #End one epoch
        #count -= 1
        print('\nepoch %5d\t learning_rate = %g\t mean_loss = %.4f\t train_acc = %.2f%%\t train_iou_acc = %.2f%%\t train_ellip_acc = %.2f' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count), (sum_acc_ellip/count)))
        print('Take time %3.1f' % (time.time() - t0))


        return step


#------------------------------------------------------------------------------

#Main function
def main():
 
    with tf.device('/gpu:0'):
        #train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        #print('The number of train records is %d and valid records is %d.' % (len(train_records), len(valid_records)))
        model = SeqFCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS,  cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob)
        model.build()
        model.train()



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

