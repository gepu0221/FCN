import numpy as np
from easydict import EasyDict as edict


cfgs = edict()
cfgs.file_list = ['training.txt', 'validation.txt']
cfgs.seq_num = 0
cfgs.cur_channel = 3

cfgs.NUM_OF_CLASSESS = 2
#cfgs.IMAGE_SIZE = [270, 480]
cfgs.IMAGE_SIZE = [540, 960]
cfgs.batch_size = 8
