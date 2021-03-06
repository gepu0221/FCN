#This code is for generating the result of prev frame after warping. 
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
from six.moves import cPickle as pickle
from six.moves import xrange
from tools.label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from tools.generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
from tools.flow_color import flowToColor, med_flow_color
from tools.data_preprocess import normal_data, concat_data
import shutil, random

#from train_seq_parent import FCNNet
from train_InpaintFlow_parent import U_Net as FCNNet

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
        self.train_optimizer()
        #self.accuracy()

        
     #6. else
    def create_view_path(self):
        train_path = os.path.join(cfgs.view_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        valid_path = os.path.join(cfgs.view_path, 'valid')
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)


    # Picle pickle and load.
    def pickle_f(self, content, fn):
        path_ = os.path.join(cfgs.pickle_path, fn+'.pickle')
        with open(path_, 'wb') as f:
            pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path_):
    
        with open(path_, 'rb') as f:
            result = pickle.load(f)

        return result
    #--------------------------------------------------------
    
    def dilate_(self, im):
        im = 255 - im[0]
        kernel = np.ones((5,5), np.uint8)
        for i in range(cfgs.dilate_num):
            im = cv2.dilate(im, kernel)
        
        im = np.expand_dims(np.expand_dims(im, axis=2), axis=0)
        
        return 255 - im

    def view_flow_one(self, flow, fn, step, a_str=''):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, 'train')
        color_flow = flowToColor(flow)
        #color_flow = med_flow_color(flow)
        cv2.imwrite(os.path.join(path_, filename+'_flow_color%s.bmp' % a_str), color_flow)
        
 
    def view_inst_mask_one(self, inst_mask, fn, step, a_str=''):
        '''
            Generate mask of instrument removed.
        '''
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn
            
        inst_mask = inst_mask[0]

        path_ = os.path.join(cfgs.view_path, 'train')
        inst_mask3 = inst_mask
        inst_mask3 = np.append(inst_mask3, inst_mask, axis=2)
        inst_mask3 = np.append(inst_mask3, inst_mask, axis=2)

        cv2.imwrite(os.path.join(path_, filename+'_inst_mask%s.bmp' % a_str), inst_mask) 
        


    def view_one(self, fn, pred_anno, pred_pro, im, step, str_a='', if_more=True):
        path_ = os.path.join(cfgs.view_path, 'train')
        
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        #pred_anno_im = (pred_anno*127).astype(np.uint8)
        #pdb.set_trace()
        prev_warping = cv2.cvtColor(pred_anno.astype(np.uint8), cv2.COLOR_BGR2RGB)
        #prev_warping = pred_anno
        cv2.imwrite(os.path.join(path_, filename+'_warp%s.bmp' % str_a), prev_warping)

        if if_more:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path_, filename+'_im%s.bmp' % str_a), im)
            im_prev = cv2.cvtColor(pred_pro, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path_, filename+'_im_prev%s.bmp' % str_a), im_prev)
        
       
            prev_diff = prev_warping - im_prev 
            #cv2.imwrite(os.path.join(path_, filename+'_diff_pw%s.bmp' % str_a), prev_diff) 
            prev_combined = im + (im_prev/2).astype(np.uint8)
            #cv2.imwrite(os.path.join(path_, filename+'_com_cp%s.bmp' % str_a), prev_combined) 
            prev_combined = im + (im_prev/2).astype(np.uint8) + (prev_warping/4).astype(np.uint8)
            #cv2.imwrite(os.path.join(path_, filename+'_com_all%s.bmp' % str_a), prev_combined) 
            warp_combined = im + (prev_warping/2).astype(np.uint8)
            #cv2.imwrite(os.path.join(path_, filename+'_com_cw%s.bmp' % str_a), warp_combined) 
            prev_warp_combined = im_prev + (prev_warping/2).astype(np.uint8)
            #cv2.imwrite(os.path.join(path_, filename+'_com_pw%s.bmp' % str_a), prev_warp_combined) 


        if cfgs.view_seq:
            for i in range(cfgs.seq_num):
                im_ = im[:,:,self.cur_channel-1+i]
                cv2.imwrite(os.path.join(path_, filename+'seq_'+str(i+1)+'.bmp'), im_)
            
    def view(self, fns, pred_annos, pred_pros, ims, step, str_a='', if_more=True):
        num_ = fns.shape[0]
        #pdb.set_trace()
        if cfgs.random_view:
            choosen = random.randint(0, num_-1)
            self.view_one(fns[choosen], pred_annos[choosen], pred_pros[choosen], ims[choosen], step, str_a, if_more)
        else:
            for i in range(num_):
                self.view_one(fns[i], pred_annos[i], pred_pros[i], ims[i], step, str_a, if_mo4e)   

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

    
    def persist_cornea(self):
        '''
            remove the part of instrutment and background to persist cornea part.
        '''
        self.persist_im = tf.placeholder(tf.float32)
        sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]]
        comp = tf.ones(sz, dtype=tf.float32)
        pro_one = tf.expand_dims(tf.where(tf.less_equal(self.unet_pro[:, :, :, 2], cfgs.low_pro), 1-comp, comp), dim=3)
        pro = pro_one
        pro = tf.concat([pro, pro_one], axis=3)
        pro = tf.concat([pro, pro_one], axis=3)
        self.corn_mask = pro_one
        self.corn_mask_im = tf.multiply(self.persist_im, pro)

    def remove_inst(self):
        '''
            remove the part of instrutment and background to persist cornea part.
        '''
        #self.persist_im = tf.placeholder(tf.float32)
        sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]]
        comp = tf.ones(sz, dtype=tf.float32)
        pro_one = tf.expand_dims(tf.where(tf.less_equal(self.unet_pro[:, :, :, 1], cfgs.low_pro), comp, 1-comp), dim=3)
        pro = pro_one
        pro = tf.concat([pro, pro_one], axis=3)
        pro = tf.concat([pro, pro_one], axis=3)
        self.no_inst_mask = pro_one
        #self.no_inst_mask_im = tf.multiply(self.persist_im, pro)
    
    def remove_flow_inst(self):
        '''
            Remove the part of instrument part of flow map.
        '''
        # Mask of reomving instrument part of current frame.
        self.flow_inst_mask = tf.placeholder(tf.float32)
        # Mask of persisting cornea part of current frame.
        self.flow_corn_mask = tf.placeholder(tf.float32)
        # Mask of removing instrument part of prev frame.
        self.flow_prev_corn_mask = tf.placeholder(tf.float32)

        no_inst_pro = tf.concat([self.flow_inst_mask, self.flow_inst_mask], axis=3)
        self.no_inst_flow = tf.multiply(self.flow_tensor, no_inst_pro)

        corn_pro = tf.concat([self.flow_corn_mask, self.flow_corn_mask], axis=3)
        corn_num = tf.reduce_sum(self.flow_corn_mask)
        self.corn_flow = tf.multiply(self.flow_tensor, corn_pro)
        self.corn_flow_mean = tf.reduce_sum(self.corn_flow, [0, 1, 2]) / corn_num
        sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 2]
        comp = tf.ones(sz, dtype=tf.float32)
        mean_comp = comp * self.corn_flow_mean
               
        self.insect_area = tf.multiply((comp - self.flow_inst_mask), self.flow_prev_corn_mask)
        self.pred_corn_flow = tf.multiply(self.insect_area, mean_comp)
        self.flow_comb = self.no_inst_flow + self.pred_corn_flow

        #self.no_inst_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.flow_comb)

    def remove_flow_inst(self):
        '''
            Remove the part of instrument part of flow map.
        '''
        # Mask of reomving instrument part of current frame.
        self.flow_inst_mask = tf.placeholder(tf.float32)
        # Mask of persisting cornea part of current frame.
        self.flow_corn_mask = tf.placeholder(tf.float32)
        # Mask of removing instrument part of prev frame.
        self.flow_prev_corn_mask = tf.placeholder(tf.float32)
        # Segmentation(logits) of prev frame to warp to another.
        self.flow_segm = tf.placeholder(tf.float32)

        no_inst_pro = tf.concat([self.flow_inst_mask, self.flow_inst_mask], axis=3)
        self.no_inst_flow = tf.multiply(self.flow_tensor, no_inst_pro)

        corn_pro = tf.concat([self.flow_corn_mask, self.flow_corn_mask], axis=3)
        corn_num = tf.reduce_sum(self.flow_corn_mask)
        self.corn_flow = tf.multiply(self.flow_tensor, corn_pro)
        self.corn_flow_mean = tf.reduce_sum(self.corn_flow, [0, 1, 2]) / corn_num
        sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 2]
        comp = tf.ones(sz, dtype=tf.float32)
        mean_comp = comp * self.corn_flow_mean
               
        self.insect_area = tf.multiply((comp - self.flow_inst_mask), self.flow_prev_corn_mask)
        self.pred_corn_flow = tf.multiply(self.insect_area, mean_comp)
        self.flow_comb = self.no_inst_flow + self.pred_corn_flow
        
        #For image
        self.no_inst_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.flow_comb)
        # For segmentation(logits)
        self.no_inst_warped_segm = self.bilinear_warping_module.bilinear_warping(self.flow_segm, self.flow_comb)
        self.no_inst_warped_segm = tf.expand_dims(tf.argmax(tf.nn.softmax(self.no_inst_warped_segm), dimension=3), axis=3)
    
    def flow_warp(self):
        
        self.flow_warp_img0 = tf.placeholder(tf.float32)
        self.flow_warp_img1 = tf.placeholder(tf.float32)

        self.warp_flow = tf.placeholder(tf.float32)
        
        self.result_warped_segm = self.bilinear_warping_module.bilinear_warping(self.flow_warp_img1, self.warp_flow)
        self.result_warped = tf.expand_dims(tf.argmax(tf.nn.softmax(self.result_warped_segm), dimension=3), axis=3)
    
        sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], 1]
        comp = tf.ones(sz, dtype=tf.float32)
        anno_img0 = tf.expand_dims(tf.argmax(tf.nn.softmax(self.flow_warp_img0), dimension=3), axis=3)
        anno_img0 = tf.cast(anno_img0 / 2, tf.int32)
        insect_area = tf.cast((tf.multiply(self.result_warped, tf.cast(comp-self.flow_inst_mask, tf.int64))) / 2, tf.int32)

        self.insect_result_warped = insect_area + anno_img0

    # For im    
    def train_one_epoch(self, sess, data_loader, data_num, epoch, step):
        
        '''
        Generate once of image which remove instruments.
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
        
        self.ellip_acc, self.accu, self.accu_iou, loss = 0, 0, 0, 0


        for count in range(1, data_num):
            step += 1
            
            images, images_pad, mask, fn = data_loader.get_next_sequence()

            # Optical flow
            optflow = []
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images[frame], images[frame-1]
                flow = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
                
                optflow.append(flow)

            
            normal_flow, normal_max_v = normal_data(flow)
            inpt_input = concat_data(normal_flow, mask, cfgs.grid)
            
            _, loss, inpt_warped_im, inpt_flow =sess.run([self.opt, self.loss, \
                                        self.inpt_warped_im, self.inpt_pred_flow],\
                                               feed_dict={self.flow_img1: last_im,
                                                          self.flow_img0: im,
                                                          self.max_v: normal_max_v,
                                                          self.inpt_data: inpt_input})
            #if epoch % 5 == 0:
            if epoch == 26:
                self.view(np.expand_dims(fn, 0), inpt_warped_im, images[cfgs.seq_frames-2], images[cfgs.seq_frames-1], step)
            
                self.view_flow_one(flow[0], fn, step)
                self.view_flow_one(inpt_flow[0], fn, step, '_inpt')

            


            #2. calculate accurary
            self.ellip_acc = 0
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

