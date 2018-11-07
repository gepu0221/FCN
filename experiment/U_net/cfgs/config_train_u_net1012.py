import numpy as np
from easydict import EasyDict as edict


cfgs = edict()
#1. path
#1.1 Image path
cfgs.file_list = ['training.txt', 'validation.txt']
cfgs.seq_list_path = '/home/gp/repos/FCN/experiment/finetune_fcn/data_info/data_list'

#1.2 Label path
#One pixel fine-tune dis4 labels.
#cfgs.anno_path = '/home/gp/repos/FCN/experiment/finetune_label/s8_dis4_1011']
#3 pixel fine-tune dis4 labels.
#cfgs.anno_path = '/home/gp/repos/FCN/experiment/finetune_label/s8_dis4_1011_dilate'
#One pixel origin labels.
#cfgs.anno_path = '/home/gp/repos/FCN/experiment/finetune_label/s8_ori_label_1013'
#Remove occulded area labels fine-tune labels.
cfgs.anno_path = '/home/gp/repos/FCN/experiment/finetune_label/s8_dis4_1011_noOcculde'
#Remove occulded area origin labels  .
#cfgs.anno_path = '/home/gp/repos/FCN/experiment/finetune_label/s8_ori_label_noOcculde1019'


#cfgs.data_pickle_fn = "data_fintune1013.pickle"
#cfgs.data_pickle_fn = "data_fintune3_1016.pickle"
cfgs.data_pickle_fn = "data_fintune3_noOccu_1101.pickle"
#cfgs.data_pickle_fn = "data_ori3_noOccu1020.pickle"


#1.3 Logs path
#cfgs.logs_dir = 'logs/logs_s8finetune1013/'
#cfgs.logs_dir = 'logs/logs_s8_ori_1015/'
#cfgs.logs_dir = 'logs/logs_focal_s8finetune1012/'
cfgs.logs_dir = 'logs/logs_s8finetune3_noOcculde_cWH_range_1025/'

#1.4 Others
cfgs.re_path ='/home/gp/repos/FCN/experiment/finetune_fcn/result_recover'
cfgs.error_path = 'ellip_error/dis4_1015'
#cfgs.error_path = 'ellip_error/ori_1016'
cfgs.shelter_pickle_path = 'key_frame_pickle/shelter_data0901.pickle'
#cfgs.view_path = 'view/s8_fintune3_noOccu_cWH_accu_e30_off0005_imMask_1029'
#cfgs.view_path = 'view/s8_ori3_noOccu1020'
#cfgs.view_path = 'view/s8_finetune1_noOccu_finegrained_1029'
cfgs.view_path = 'view/s8_fintune3_noOccu_cWH_accu_e30_update_1101'


#2. param
cfgs.seq_num = 0
cfgs.cur_channel = 3
cfgs.batch_size = 8
cfgs.keep_prob = 0.5
cfgs.input_keep_prob = 0.8
cfgs.train_itr = 0
cfgs.learning_rate_path = 'lr_f.txt'
cfgs.init_lr = 0.01
cfgs.model_dir = '/home/gp/repos/FCN/Model_zoo/Resnet'
cfgs.MODEL_NAME = "imagenet-resnet-101-dag.mat"
cfgs.result_dir = 'result/'
cfgs.debug = 'False'
cfgs.mode = 'train'
cfgs.max_epochs = 31
cfgs.NUM_OF_CLASSESS = 2
#cfgs.IMAGE_SIZE = [270, 480]
cfgs.IMAGE_SIZE = [540, 960]
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

#3. Resnet
cfgs.num_units_list = [3, 4, 23, 3]
cfgs.first_stride_list = [1, 2, 2, 2]
cfgs.depth_list = [256, 512, 1024, 2048]

#lower probability
cfgs.low_pro = 0.1
cfgs.offset = [0, 1]
#cfgs.offset = [0, 0.2]
#4. vis choices
#cfgs.mode = 'visualize'
#cfgs.mode = 'vis_video'
cfgs.anno = True
cfgs.heatmap = False
cfgs.trans_heat = True
cfgs.fit_ellip = True
cfgs.anno_fuse = False
cfgs.fuse_ellip = False
cfgs.lower_anno = False
cfgs.fit_ellip_lower = False
cfgs.view_seq = False
cfgs.random_view = False
cfgs.test_accu = True
cfgs.test_view = True
cfgs.if_train = True

#5. generate result data
#cfgs.mode = 'generate_re'

#6 Hausdorrf loss'
#cfgs.mode = 'haus_loss'
cfgs.eps = 1e-6
cfgs.alpha = 4
