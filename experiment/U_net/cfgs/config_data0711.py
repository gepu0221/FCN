import numpy as np
from easydict import EasyDict as edict


cfgs = edict()
cfgs.file_list = ['train_seq_list.txt', 'validation_seq_list.txt']
cfgs.seq_num = 4
cfgs.seq_list_path = '/home/gp/repos/data/seq_list'
cfgs.anno_path = '/home/gp/repos/FCN/annotation_seq0727_bmp1'

cfgs.batch_size = 32
cfgs.keep_prob = 0.5
cfgs.v_batch_size = 12
cfgs.logs_dir = 'logs_seq20180727_total/'
cfgs.train_itr = 1
cfgs.data_dir = 'image_save20180517_finetune_017'
cfgs.learning_rate_path = 'lr_f.txt'
cfgs.model_dir = '/home/gp/repos/FCN/Model_zoo/'
cfgs.result_dir = 'result/'
cfgs.debug = True
cfgs.mode = 'train'
cfgs.max_epochs = 500
cfgs.NUM_OF_CLASSESS = 2
cfgs.MAX_ITERATION = 100000
cfgs.IMAGE_SIZE = 224
cfgs.init_lr = 0.01

#focal
cfgs.at = 2
cfgs.gamma = 0.25

#vis
#cfgs.mode = 'visualize'
#cfgs.mode = 'video_vis'
cfgs.anno = True
cfgs.heatmap = True
cfgs.trans_heat = False
cfgs.fit_ellip = True


