import numpy as np
from easydict import EasyDict as edict


cfgs = edict()
#1. path
#1.1 Image path
cfgs.image_path = '/home/gp/repos/FCN/experiment/data_process/data/inst_da/s8_video_part_3000_noInst'
cfgs.da_im_path = '/home/gp/repos/FCN/experiment/data_process/data/inst_da/rect_inst_da1215'

#1.2 Label path
#Remove occulded area labels fine-tune labels.
cfgs.train_anno_path = '/home/gp/repos/FCN/experiment/data_process/data/s8_part_video_gtFine3_inter6_1206/train'
cfgs.val_anno_path = '/home/gp/repos/FCN/experiment/data_process/data/s8_part_video_gtFine3_inter6_1206/val'

# inpainting flow mask path
cfgs.train_mask_path = '/home/gp/repos/FCN/experiment/data_process/data/inst_da/rect_inst_da_mask1215/train'
cfgs.val_mask_path = '/home/gp/repos/FCN/experiment/data_process/data/inst_da/rect_inst_da_mask1215/val'

# Load inpainting flow pickle.
cfgs.inpt_flow_pickle = '/home/gp/repos/FCN/experiment/GRU/data/inpt_flow_dilate_pickle'


#1.3 Logs path
cfgs.unet_logs_dir = '/home/gp/repos/FCN/experiment/weakly_supervised/logs/logs_s8ori3_noOcculde_label2_1116/'
cfgs.flow_logs_dir = 'logs/flow/'
cfgs.flow_logs_name = 'flownet2'
#I 10: inter 10
#rsz: resize
#warpAdd: add warp loss to loss function
cfgs.gru_logs_dir = 'logs/s8_inpt_RectInstDa_I10_rsz256_warpAdd05_1219/'


#1.4 Others
#cfgs.view_path = 'view/s8_gru_flow_inpt_warp1210'
cfgs.view_path = 'view_inpt/s8_RectInptDa_I10_rsz256_warpAdd05_1219'
cfgs.pickle_path = 'data/flow_dilate_pickle'

#2. unet param
cfgs.class_convs_num = 2 
cfgs.p_k_sz = [3, 4] 

cfgs.seq_num = 0
cfgs.cur_channel = 3
cfgs.n_class = 3
cfgs.layers = 5
cfgs.features_root = 64
cfgs.filter_size = 3
cfgs.pool_size = 2
cfgs.batch_size = 1
cfgs.keep_prob = 1
cfgs.input_keep_prob = 0.8
cfgs.train_itr = 0
cfgs.learning_rate_path = 'lr_f.txt'
cfgs.init_lr = 0.01
cfgs.model_dir = '/home/gp/repos/FCN/Model_zoo/Resnet'
cfgs.MODEL_NAME = "imagenet-resnet-101-dag.mat"
cfgs.result_dir = 'result/'
cfgs.debug = 'False'
cfgs.mode = 'train'
cfgs.max_epochs = 11
cfgs.NUM_OF_CLASSESS = 3

cfgs.IMAGE_SIZE = [270, 480]
#cfgs.IMAGE_SIZE = [540, 960]
cfgs.if_pad = [False, False, True, True]
cfgs.pad_num_w = 0
cfgs.pad_num_h = 1
#cfgs.mean_pixel = np.array([103.939, 116.779, 123.68, 117.378, 117.378, 117.378, 117.378])
#cfgs.mean_pixel = np.array([117.378, 117.378, 117.378, 117.378, 117.378])
#cfgs.mean_pixel = np.array([103.939, 116.779, 123.68, 0,0,0,0,0,0,0,0,0,0])
cfgs.mean_pixel = np.array([103.939, 116.779, 123.68])

#center_loss
#cfgs.center_w = 0.00001
cfgs.center_w = 0.00001
cfgs.dis_w = 0.0002

#focal
cfgs.at = 0.25
cfgs.gamma = 2

#lower probability
cfgs.low_pro = 0.5
cfgs.inst_low_pro = 0


#3. GRU
cfgs.grfp_lr = 2e-5
cfgs.train_num = 2236
cfgs.valid_num = 248
#----
cfgs.seq_frames = 2
cfgs.nbr_frames = 2
#----
cfgs.inter = 10
cfgs.frame = 10617
#---
cfgs.if_dilate = True
cfgs.dilate_num = 5

#4. Inpainting
cfgs.inpt_lr = 1e-5
cfgs.inpt_in_channel = 2
cfgs.grid = 8
cfgs.grid_padding = 0
cfgs.inpt_resize_im_sz = [256, 256]
cfgs.inpt_resize = True
cfgs.w_warp_loss = 0.5

#5. vis choices
cfgs.anno = True
cfgs.heatmap = False
cfgs.trans_heat = True
cfgs.fit_ellip = True
cfgs.anno_fuse = False
cfgs.fuse_ellip = False
cfgs.lower_anno = False
cfgs.fit_ellip_lower = False
cfgs.view_seq = False
cfgs.random_view = True
cfgs.test_accu = True
cfgs.test_view = True
cfgs.if_train = True


#6 Hausdorrf loss'
#cfgs.mode = 'haus_loss'
cfgs.eps = 1e-6
cfgs.alpha = 4
