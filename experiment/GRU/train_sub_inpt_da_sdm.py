#This code 
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
from tools.data_preprocess import normal_data, concat_data, sdm, local_patch, expand_sdm, expand_local_patch
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
        #self.expand_sdm_loss()
        self.loss()
        self.error_acu()
        #self.poly_loss()
        self.train_optimizer()

        
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

    def view_flow_one(self, flow, fn, step, a_str='', f_path='train'):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, f_path)
        color_flow = flowToColor(flow)
        if cfgs.inpt_resize:
            h, w = cfgs.IMAGE_SIZE
            color_flow = cv2.resize(color_flow, (w, h), interpolation=cv2.INTER_CUBIC)
        #color_flow = med_flow_color(flow)
        cv2.imwrite(os.path.join(path_, filename+'_flow_%s.bmp' % a_str), color_flow)
        
    def view_flow_patch_one(self, flow, fn, step, a_str='', f_path='train'):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, f_path)
        color_flow = flowToColor(flow)
        h, w = cfgs.IMAGE_SIZE
        #color_flow = cv2.resize(color_flow, (w, h), interpolation=cv2.INTER_CUBIC)
        #color_flow = med_flow_color(flow)
        cv2.imwrite(os.path.join(path_, filename+'_flow_%s.bmp' % a_str), color_flow)
    
    def view_patch_one(self, patch, fn, step, a_str='', f_path='train'):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, f_path)
        #color_flow = cv2.resize(color_flow, (w, h), interpolation=cv2.INTER_CUBIC)
        #color_flow = med_flow_color(flow)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path_, filename+'_flow_%s.bmp' % a_str), patch)


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
        


    def view_one(self, fn, pred_anno, pred_pro, im, step, str_a='', if_more=True, f_path='train'):
        path_ = os.path.join(cfgs.view_path, f_path)
        
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn
        h, w = cfgs.IMAGE_SIZE
        prev_warping = cv2.cvtColor(pred_anno.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if cfgs.inpt_resize:
            prev_warping = cv2.resize(prev_warping, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(path_, filename+'_warp%s.bmp' % str_a), prev_warping)
        
        if if_more:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
            if cfgs.inpt_resize:
                cv2.imwrite(os.path.join(path_, filename+'_im%s.bmp' % str_a), im)
            im_prev = cv2.cvtColor(pred_pro, cv2.COLOR_BGR2RGB)
            if cfgs.inpt_resize:
                im_prev = cv2.resize(im_prev, (w, h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(path_, filename+'_im_prev%s.bmp' % str_a), im_prev)
        
    def view(self, fns, pred_annos, pred_pros, ims, step, str_a='', if_more=True, f_path='train'):
        num_ = fns.shape[0]
        if cfgs.random_view:
            choosen = random.randint(0, num_-1)
            self.view_one(fns[choosen], pred_annos[choosen], pred_pros[choosen], ims[choosen], step, str_a, if_more, f_path)
        else:
            for i in range(num_):
                self.view_one(fns[i], pred_annos[i], pred_pros[i], ims[i], step, str_a, if_more, f_path)   

        
        
    def loss(self):
        
        # 1. warped loss
        self.ori_flow = tf.placeholder(tf.float32)
        self.inpt_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.inpt_pred_flow)
        self.ori_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.ori_flow)
        self.im_warp_loss = tf.reduce_mean(tf.abs(self.inpt_warped_im - self.ori_warped_im))
        
        # 2. mask optical loss
        batch_raw, masks_raw = tf.split(self.inpt_data, 2, axis=2)
        mask = tf.cast(masks_raw[0:1, :, :, 0:2] > 127.5, tf.float32)
        paddings = tf.constant([[0, 0], [0, cfgs.grid_padding], [0, 0], [0, 0]])
        mask = tf.pad(mask, paddings, 'CONSTANT')

        self.rect_param = tf.placeholder(shape = [4], dtype=tf.int32)
        self.sd_mask = tf.placeholder(tf.float32)
        local_patch_inpt_flow = local_patch(self.inpt_pred_flow, self.rect_param)
        local_patch_ori_flow = local_patch(self.ori_flow, self.rect_param)
        self.local_patch_inpt_flow = local_patch_inpt_flow


        self.flow_loss = tf.reduce_mean(tf.abs(local_patch_inpt_flow - local_patch_ori_flow) * self.sd_mask)
        #self.flow_loss = tf.reduce_mean(tf.abs(local_patch_inpt_flow - local_patch_ori_flow))

        self.flow_loss_ = tf.reduce_mean(tf.abs(local_patch_inpt_flow - local_patch_ori_flow))


        # 3. Use prediction complete flow to caculate loss.
        self.complete_flow_loss = tf.reduce_mean(tf.abs(self.ori_flow - self.pred_complete_flow))

        #self.loss = self.complete_flow_loss
        self.loss = self.flow_loss
        #self.loss = cfgs.w_warp_loss * self.im_warp_loss + (1-cfgs.w_warp_loss) * self.flow_loss
    
    def error_acu(self):
        #flow
        self.ori_flow_normal, _ = tf.split(self.inpt_data, 2, axis=2)
        # ori_im
        self.ori_cur_im = tf.placeholder(tf.float32)

        #self.rect_param = tf.placeholder(shape = [4], dtype=tf.int32)
        local_inpt_flow_normal = local_patch(self.inpt_pred_flow_normal, self.rect_param)
        local_ori_flow_normal = local_patch(self.ori_flow_normal, self.rect_param)

        self.l1_e =tf.reduce_mean(tf.abs(local_inpt_flow_normal - local_ori_flow_normal)) / 2
        self.l2_e = tf.reduce_mean((local_inpt_flow_normal - local_ori_flow_normal)**2 ) / 4

        #warped im 
        local_ori_warped_im = local_patch(self.ori_warped_im, self.rect_param)
        local_inpt_warped_im = local_patch(self.inpt_warped_im, self.rect_param)

        self.w_l1_e =tf.reduce_mean(tf.abs(local_ori_warped_im - local_inpt_warped_im)) / 256
        self.w_l2_e = tf.reduce_mean((local_ori_warped_im/256 - local_inpt_warped_im/256)**2 )
        
        self.w_psnr = -10.0 * tf.log(tf.cast(tf.reduce_mean(((local_ori_warped_im-local_inpt_warped_im)/256)**2), tf.float32)) / tf.log(10.0)



        #im
        local_ori_im = local_patch(self.ori_cur_im, self.rect_param)
        local_warped_im = local_patch(self.inpt_warped_im, self.rect_param)
        self.local_ori_im = local_ori_im
        self.local_warped_im = local_warped_im

        self.im_l1_e =tf.reduce_mean(tf.abs(local_ori_im - local_warped_im)) / 256
        self.im_l2_e = tf.reduce_mean((local_ori_im/256 - local_warped_im/256)**2 )

        self.psnr = -10.0 * tf.log(tf.cast(tf.reduce_mean(((local_ori_im-local_warped_im)/256)**2), tf.float32)) / tf.log(10.0)


    # Expand sdm loss
    def expand_sdm_loss(self):
        
        # 1. warped loss
        self.ori_flow = tf.placeholder(tf.float32)
        self.inpt_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.inpt_pred_flow)
        self.ori_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.ori_flow)
        self.im_warp_loss = tf.reduce_mean(tf.abs(self.inpt_warped_im - self.ori_warped_im))
        
        # 2. mask optical loss
        batch_raw, masks_raw = tf.split(self.inpt_data, 2, axis=2)
        mask = tf.cast(masks_raw[0:1, :, :, 0:2] > 127.5, tf.float32)
        paddings = tf.constant([[0, 0], [0, cfgs.grid_padding], [0, 0], [0, 0]])
        mask = tf.pad(mask, paddings, 'CONSTANT')

        self.rect_param = tf.placeholder(shape = [4], dtype=tf.int32)
        self.epd_rect_param = tf.placeholder(shape = [4], dtype=tf.int32)
        self.epd_sd_mask = tf.placeholder(tf.float32)

        local_patch_inpt_flow = local_patch(self.inpt_pred_flow, self.rect_param)
        local_patch_ori_flow = local_patch(self.ori_flow, self.rect_param)
        self.local_patch_inpt_flow = local_patch_inpt_flow
        
        epd_l_p_inpt_flow = local_patch(self.inpt_pred_flow, self.epd_rect_param)
        epd_l_p_ori_flow = local_patch(self.ori_flow, self.epd_rect_param)


        self.flow_loss = tf.reduce_mean(tf.abs(epd_l_p_inpt_flow - epd_l_p_ori_flow) * self.epd_sd_mask)
        #self.flow_loss = tf.reduce_mean(tf.abs(local_patch_inpt_flow - local_patch_ori_flow))

        self.flow_loss_ = tf.reduce_mean(tf.abs(local_patch_inpt_flow - local_patch_ori_flow))


        # 3. Use prediction complete flow to caculate loss.
        self.complete_flow_loss = tf.reduce_mean(tf.abs(self.ori_flow - self.pred_complete_flow))

        #self.loss = self.complete_flow_loss
        #self.loss = self.flow_loss
        self.loss = cfgs.w_warp_loss * self.im_warp_loss + (1-cfgs.w_warp_loss) * self.flow_loss


    
    def poly_loss(self):
        
        self.ori_flow = tf.placeholder(tf.float32)
        
        self.inpt_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.inpt_pred_flow)
        self.ori_warped_im = self.bilinear_warping_module.bilinear_warping(self.flow_img1, self.ori_flow)
        self.im_warp_loss = tf.reduce_mean(tf.abs(self.inpt_warped_im - self.ori_warped_im))
        
        batch_raw, masks_raw = tf.split(self.inpt_data, 2, axis=2)
        mask = tf.cast(masks_raw[0:1, :, :, 0:2] > 127.5, tf.float32)
        paddings = tf.constant([[0, 0], [0, cfgs.grid_padding], [0, 0], [0, 0]])
        mask = tf.pad(mask, paddings, 'CONSTANT')

        self.mask_sum = tf.reduce_sum(mask)
        self.flow_loss = tf.reduce_sum(tf.abs(self.ori_flow - self.inpt_pred_flow) * mask) / self.mask_sum

        #Use prediction complete flow to caculate loss.
        self.complete_flow_loss = tf.reduce_mean(tf.abs(self.ori_flow - self.pred_complete_flow))

        #self.loss = self.complete_flow_loss
        #self.loss = self.flow_loss
        self.loss = cfgs.w_warp_loss * self.im_warp_loss + (1-cfgs.w_warp_loss) * self.flow_loss


    # For im    
    def train_one_epoch(self, sess, data_loader, data_num, epoch, step):
        
        '''
        Generate once of image which remove instruments.
        Args:
            data_loader: training or validation data_loader.
            data_num: number of data.
        '''
        total_loss, total_im_warp_loss, total_flow_loss = 0, 0, 0
        count = 0
        t0 =time.time()

        im_warp_loss, flow_loss  = 0, 0
        
        #True number to loss
        t_count = 0

        for count in range(1, data_num):
            step += 1
            
            images, images_da, mask, fn, flag, rect_param = data_loader.get_next_sequence()
            if flag == False:
                #print(fn)
                # Can't find prev data.
                continue
            # Origin optical flow
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images[frame], images[frame-1]
                flow = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
            # Da optical flow
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images_da[frame], images_da[frame-1]
                flow_da = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
                

            normal_flow, normal_max_v = normal_data(flow)
            inpt_input, flag_ = concat_data(normal_flow, mask, cfgs.grid)
            sd_mask = sdm(rect_param, cfgs.gamma)
            #epd_sd_mask, epd_rect_param = expand_sdm(rect_param, cfgs.gamma, cfgs.epd_ratio)


            if flag_ == False:
                print(fn)
                #After grid, area of mask = 0
                continue
            
            t_count += 1
            _, loss, im_warp_loss, flow_loss, \
            inpt_warped_im, inpt_flow, pred_flow, \
             =sess.run([\
             self.opt,\
             self.loss, self.im_warp_loss, self.flow_loss_, \
                                        self.inpt_warped_im, self.inpt_pred_flow, self.pred_complete_flow],\
                                               feed_dict={self.flow_img1: last_im,
                                                          self.flow_img0: im,
                                                          self.ori_flow: flow,
                                                          self.max_v: normal_max_v,
                                                          self.inpt_data: inpt_input,
                                                          #self.epd_sd_mask: epd_sd_mask,
                                                          #self.epd_rect_param: epd_rect_param,
                                                          self.sd_mask: sd_mask,
                                                          self.rect_param: rect_param})
            
            
            if count % 200 == 0:
                self.view(np.expand_dims(fn, 0), inpt_warped_im, images[cfgs.seq_frames-2], images[cfgs.seq_frames-1], step)
                self.view(np.expand_dims(fn, 0), inpt_warped_im, images_da[cfgs.seq_frames-2], images_da[cfgs.seq_frames-1], step), '_da'
                self.view_flow_one(flow[0], fn, step)
                self.view_flow_one(pred_flow[0], fn, step, 'complete')
                self.view_flow_one(inpt_flow[0], fn, step, 'inpt')
                self.view_flow_one(flow_da[0], fn, step, 'da')
            #3. calculate loss
            total_loss += loss
            total_im_warp_loss += im_warp_loss
            total_flow_loss += flow_loss

            #4. time consume
            time_consumed = time.time() - t0
            time_per_batch = time_consumed/count

            #5. print
            line = 'Train epoch %2d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t m_loss=%.4f\t  m_imW_loss = %.4f\t m_f_loss = %.4f\t  time = %.2f' % (epoch, cfgs.inpt_lr, step, count, loss, (total_loss/t_count), (total_im_warp_loss/t_count), (total_flow_loss/t_count), time_per_batch)
            utils.clear_line(len(line))
            print('\r' + line, end='')

        #End one epoch
        #count -= 1
        print('\nepoch %5d\t learning_rate = %g\t mean_loss = %.4f\t m_imW_loss = %.4f\t m_f_loss = %.4f\t ' % (epoch, cfgs.inpt_lr, (total_loss/t_count), (total_im_warp_loss/t_count), (total_flow_loss/t_count)))
        print('Take time %3.1f' % (time.time() - t0))


        return step
    
    
    def valid_one_epoch(self, sess, data_loader, data_num, epoch, step):
        
        '''
        Generate once of image which remove instruments.
        Args:
            data_loader: training or validation data_loader.
            data_num: number of data.
        '''
        total_loss, total_im_warp_loss, total_flow_loss = 0, 0, 0
        total_l1, total_im_l1, total_l2, total_im_l2, total_psnr = 0, 0, 0, 0, 0
        count = 0
        t0 =time.time()

        im_warp_loss, flow_loss  = 0, 0
        
        #True number to loss
        t_count = 0


        for count in range(1, data_num):
            step += 1
            
            images, images_da, mask, fn, flag, rect_param = data_loader.get_next_sequence_valid()
            if flag == False:
                #print(fn)
                # Can't find prev data.
                continue
            # Origin optical flow
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images[frame], images[frame-1]
                flow = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
            # Da optical flow
            for frame in range(1, cfgs.seq_frames):
                im, last_im = images_da[frame], images_da[frame-1]
                flow_da = sess.run(self.flow_tensor,
                                feed_dict={self.flow_img0: im,
                                           self.flow_img1: last_im})
                

            normal_flow, normal_max_v = normal_data(flow)
            inpt_input, flag_ = concat_data(normal_flow, mask, cfgs.grid)
            #epd_sd_mask, epd_rect_param = expand_sdm(rect_param, cfgs.gamma, cfgs.epd_ratio)
            sd_mask = sdm(rect_param, cfgs.gamma)
            if flag_ == False:
                print(fn)
                #After grid, area of mask = 0
                continue


            t_count += 1   
            loss, im_warp_loss, flow_loss,\
            l_inpt_p, \
            l1_e, l2_e, im_l1_e, im_l2_e, psnr,\
            l_ori_im, l_warped_im,\
            inpt_warped_im, inpt_flow, pred_flow \
            =sess.run([self.loss, self.im_warp_loss, self.flow_loss_,\
                       self.local_patch_inpt_flow, \
                       self.l1_e, self.l2_e, self.w_l1_e, self.w_l2_e, self.w_psnr,\
                       self.local_ori_im, self.local_warped_im,\
                       self.inpt_warped_im, self.inpt_pred_flow, self.pred_complete_flow],\
                                               feed_dict={self.flow_img1: last_im,
                                                          self.flow_img0: im,
                                                          self.ori_cur_im: images[1],
                                                          self.ori_flow: flow,
                                                          self.max_v: normal_max_v,
                                                          self.inpt_data: inpt_input,
                                                          #self.epd_sd_mask: epd_sd_mask,
                                                          #self.epd_rect_param: epd_rect_param,
                                                          self.sd_mask: sd_mask,
                                                          self.rect_param: rect_param})

            #if count % 10 == 0:
            if False:
                self.view(np.expand_dims(fn, 0), inpt_warped_im, images_da[cfgs.seq_frames-2], images_da[cfgs.seq_frames-1], step, f_path='valid')
                
                self.view_patch_one(l_ori_im[0], fn, step, 'l_ori', f_path='valid')
                self.view_patch_one(l_warped_im[0], fn, step, 'l_warped', f_path='valid')
                self.view_flow_patch_one(l_inpt_p[0], fn, step, 'l_inpt_p', f_path='valid')
                self.view_flow_one(flow[0], fn, step, f_path='valid')
                self.view_flow_one(inpt_flow[0], fn, step, '_inpt', 'valid')
                self.view_flow_one(flow_da[0], fn, step, '_da', 'valid')
                self.view_flow_one(pred_flow[0], fn, step, 'complete', f_path='valid')


            


            #3. calculate loss
            total_loss += loss
            total_im_warp_loss += im_warp_loss
            total_flow_loss += flow_loss
            total_l1 += l1_e
            total_im_l1 += im_l1_e
            total_l2 += l2_e
            total_im_l2 += im_l2_e
            total_psnr += psnr



            #4. time consume
            time_consumed = time.time() - t0
            time_per_batch = time_consumed/count

            #5. print
            #line = 'Valid epoch %2d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t m_loss=%.4f\t m_imW_loss = %.4f\t m_f_loss = %.4f\t time = %.2f' % (epoch, cfgs.inpt_lr, step, count, loss, (total_loss/t_count), (total_im_warp_loss/t_count), (total_flow_loss/t_count), time_per_batch)
            line = 'Valid epoch %2d\t step = %4d\t count = %4d\t m_imW_loss = %.2f\t m_f_loss = %.2f\t l1_e = %.3f\t im_l1_e = %.3f\t l2_e = %.3f\t im_l2_e = %.3f\t psnr = %4f\t' % (epoch, step, count, (total_im_warp_loss/t_count), (total_flow_loss/t_count), (total_l1/count), (total_im_l1/count),  (total_l2/count), (total_im_l2/count), (total_psnr/count))
 
            utils.clear_line(len(line))
            print('\r' + line, end='')

        #End one epoch
        #count -= 1
        print('\nepoch %5d\t learning_rate = %g\t mean_loss = %.4f\t m_imW_loss = %.4f\t m_f_loss = %.4f\t ' % (epoch, cfgs.inpt_lr, (total_loss/t_count), (total_im_warp_loss/t_count), (total_flow_loss/t_count)))
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

