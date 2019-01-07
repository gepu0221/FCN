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
from tools.data_preprocess import normal_data, concat_data, sdm, local_patch
import shutil, random

#from train_seq_parent import FCNNet
from train_InpaintFlow_parent_reconst import U_Net as FCNNet

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
        #self.loss()
        #self.reconst_warped_im()
        #self.poly_loss()
        #self.train_optimizer()
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

    def view_flow_one(self, flow, fn, step, a_str='', f_path='train'):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, f_path)
        color_flow = flowToColor(flow)
        if cfgs.inpt_resize:
            h, w = cfgs.IMAGE_SIZE
            #color_flow = cv2.resize(color_flow, (w, h), interpolation=cv2.INTER_CUBIC)
        #color_flow = med_flow_color(flow)
        cv2.imwrite(os.path.join(path_, filename+'_flow_%s.bmp' % a_str), color_flow)
        
    def view_im_one(self, patch, fn, step, a_str='', f_path='train'):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, f_path)
        cv2.imwrite(os.path.join(path_, filename+'_%s.bmp' % a_str), patch)

    def view_im_cvt_one(self, patch, fn, step, a_str='', f_path='train'):
        if cfgs.test_view:
            filename = fn
        else:
            filename = str(step)+'_'+fn

        path_ = os.path.join(cfgs.view_path, f_path)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path_, filename+'_%s.bmp' % a_str), patch)

         
 
    def valid_one_epoch_warped_im(self, sess, data_loader, data_num, epoch, step):
        
        '''
        Using warped prev image to do new segmentation.
        Args:
            data_loader: training or validation data_loader.
            data_num: number of data.
        '''
        total_loss, total_im_warp_loss, total_flow_loss = 0, 0, 0
        count = 0
        t0 =time.time()

        im_warp_loss, flow_loss, loss  = 0, 0, 0
        
        #True number to loss
        t_count = 0

        for count in range(1, data_num):
            step += 1
            
            #1. Load data.
            cur_im, prev_im, ellip_info, fn= data_loader.get_next_sequence()
            #if flag == False:
                #print(fn)
                # Can't find prev data.
                #continue
            

            feed_dict = {self.prev_img: prev_im,
                         self.cur_img: cur_im,
            }
            '''
            inpt_flow, inpt_warped_im,\
            cur_anno, reconst_anno, warped_anno = sess.run(\
                            [self.inpt_pred_flow, self.warped_prev_im,\
                            self.cur_static_anno_pred, self.reconst_cur_anno, self.warped_static_anno_pred],\
                                                  feed_dict=feed_dict)
            '''
            #reconst_anno = sess.run(self.reconst_cur_anno, feed_dict=feed_dict)
            reconst_anno, warped_anno, warped_im, inpt_flow, flow = sess.run([self.reconst_cur_anno, self.warped_static_anno_pred, self.warped_prev_im, self.inpt_pred_flow_re, self.inpt_pred_flow], feed_dict=feed_dict)
            self.view_im_cvt_one(warped_im[0], fn, step, 'warped_im', 'valid')
            self.view_im_one(reconst_anno[0, :, :, 1]*127, fn, step, 'anno_reconst', 'valid')
            self.view_im_one(warped_anno[0, :, :, 1]*127, fn, step, 'anno_warp', 'valid')
            self.view_flow_one(inpt_flow[0], fn, step, 'flow', 'valid' )
            self.view_flow_one(flow[0], fn, step, 'flow_no', 'valid')

            #self.view_flow_one(inpt_flow[0], fn, step, 'inpt', 'valid')
            #self.view_patch_one(inpt_warped_im[0], fn, step, 'im', 'valid')
            '''
            self.view_im_one(cur_anno[0]*127, fn, step, 'anno_cur', 'valid')
            self.view_im_one(warped_anno[0]*127, fn, step, 'anno_warped', 'valid')
            self.view_im_one(reconst_anno[0]*255, fn, step, 'anno_reconst', 'valid')
            self.view_im_cvt_one(images_pad[0][0]+inpt_warped_im[0], fn, step, 'im_prev', 'valid')
            self.view_im_cvt_one(images_pad[1][0]+inpt_warped_im[0], fn, step, 'im_cur', 'valid')
            self.view_im_cvt_one(inpt_warped_im[0], fn, step, 'im_inpt', 'valid')
            self.view_im_one(cur_anno[0]*127, fn, step, 'anno_cur', 'valid')
            '''
            

            t_count += 1
            #3. calculate loss
            total_loss += loss
            total_im_warp_loss += im_warp_loss
            total_flow_loss += flow_loss

            #4. time consume
            time_consumed = time.time() - t0
            time_per_batch = time_consumed/count

            #5. print
            #line = 'none'
            line = 'Train epoch %2d\t lr = %g\t step = %4d\t count = %4d\t loss = %.4f\t m_loss=%.4f\t  m_imW_loss = %.4f\t m_f_loss = %.4f\t  time = %.2f' % (epoch, cfgs.inpt_lr, step, count, loss, (total_loss/t_count), (total_im_warp_loss/t_count), (total_flow_loss/t_count), time_per_batch)
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

