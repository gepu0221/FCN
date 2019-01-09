import numpy as np
from easydict import EasyDict as edict

cfgs = edict()
#ellipse error metric
cfgs.loss_thresh = 50
cfgs.small_area_num = 100
cfgs.expand_r = 1.05
cfgs.re_sz = [256, 256]

cfgs.ratio = 1
cfgs.restore_sz = [540, 960]
#cfgs.error_path = 'ellip_error/s8_ori3_noOcclu_label1_validShow270_480_0104'
cfgs.error_path = 'ellip_error/s8_seq2_reoncst_remove100_0109'
cfgs.shelter_pickle_path = 'key_frame_pickle/shelter_data0901.pickle'
#Full im path
#cfgs.full_im_path = '/home/gp/repos/FCN/experiment/finetune_label/s8/Image/'
#cfgs.full_im_path = '/home/gp/repos/FCN/experiment/weakly_supervised/data/Ori_ImMask/s8_ori1_noOcclu_label2_rect_remove300_ori1124/'
cfgs.full_im_path = 'data/s8/Image'

#data list save path
cfgs.list_save_path = '/home/gp/repos/FCN/experiment/finetune_fcn/data_info/video_data_list/val_gen_1129.txt'
