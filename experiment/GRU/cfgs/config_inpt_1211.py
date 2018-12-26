import numpy as np
from easydict import EasyDict as edict


cfgs = edict()
#1. path
#1.1 Image path
cfgs.image_path = '/home/gp/repos/FCN/experiment/data_process/s8_video_part/'


#1.2 Label path
cfgs.train_anno_path = '/home/gp/repos/FCN/experiment/data_process/data/s8_part_video_gtFine3_inter6_1206/train'
cfgs.val_anno_path = '/home/gp/repos/FCN/experiment/data_process/data/s8_part_video_gtFine3_inter6_1206/val'

# Load inpainting flow pickle.
cfgs.inpt_flow_pickle = '/home/gp/repos/FCN/experiment/GRU/data/inpt_flow_dilate_pickle'


#1.3 Logs path
cfgs.unet_logs_dir = '/home/gp/repos/FCN/experiment/weakly_supervised/logs/logs_s8ori3_noOcculde_label2_1116/'
cfgs.flow_logs_dir = 'logs/flow/'
cfgs.flow_logs_name = 'flownet2'
cfgs.gru_logs_dir = 'logs/gru_nouse_1208/'

#1.4 Others
cfgs.view_path = 'view/s8_gru_flow_inpt_insect1211'
cfgs.pickle_path = 'data/flow_dilate_pickle'

#2. inpainting

#3. param
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
cfgs.max_epochs = 2
cfgs.NUM_OF_CLASSESS = 3
#cfgs.ANNO_IMAGE_SIZE = [540, 960]
#cfgs.RESIZE_IMAGE_SIZE = [540, 960]
cfgs.IMAGE_SIZE = [540, 960]
cfgs.if_pad = [False, False, True, True]
cfgs.pad_num_w = 0
cfgs.pad_num_h = 1
#cfgs.mean_pixel = np.array([103.939, 116.779, 123.68, 117.378, 117.378, 117.378, 117.378])
#cfgs.mean_pixel = np.array([117.378, 117.378, 117.378, 117.378, 117.378])
#cfgs.mean_pixel = np.array([103.939, 116.779, 123.68, 0,0,0,0,0,0,0,0,0,0])
cfgs.mean_pixel = np.array([103.939, 116.779, 123.68])


#lower probability
cfgs.low_pro = 0.5
cfgs.inst_low_pro = 0


#3. GRU
cfgs.grfp_lr = 2e-5
cfgs.train_num = 873
cfgs.valid_num = 97
#----
cfgs.seq_frames = 2
cfgs.nbr_frames = 2
#----
cfgs.inter = 6
cfgs.frame = 10617
#---
cfgs.if_dilate = True
cfgs.dilate_num = 5



#4. vis choices
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
