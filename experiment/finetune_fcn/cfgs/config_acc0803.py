import numpy as np
from easydict import EasyDict as edict

cfgs = edict()
#ellipse error metric
cfgs.loss_thresh = 1
#cfgs.error_path = 'ellip_error/s8_dis4_finetune_1015'
cfgs.error_path = 'ellip_error/s8_fintune3_noOccu_epoch10_1018'
cfgs.shelter_pickle_path = 'key_frame_pickle/shelter_data0901.pickle'
