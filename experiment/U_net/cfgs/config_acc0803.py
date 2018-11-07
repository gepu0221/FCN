import numpy as np
from easydict import EasyDict as edict

cfgs = edict()
#ellipse error metric
cfgs.loss_thresh = 1
#cfgs.error_path = 'ellip_error/s8_dis4_finetune_1015'
#cfgs.error_path = 'ellip_error/s8_fintune3_noOccu_cWH_accu_e30_update_1101'
#cfgs.error_path = 'ellip_error/s8_ori1_noOccu_imMaskGradOff0005_accu_e23_low005_1104'
#cfgs.error_path = 'ellip_error/s8_fintune1_noOccu_imMaskOff0005_1030'
#cfgs.error_path = 'ellip_error/s8_ori3_noOccu_1020'
#cfgs.error_path = 'ellip_error/s8_ori1_noOccu_imMaskOff0005_1031'
#cfgs.error_path = 'ellip_error/s8_ori1_noOccu_imMaskOff0005_e20_low01_updateGT_new_1101'
cfgs.error_path = 'ellip_error/s8_ori_noOccu_soft20_imMask06_accu_20_low01_valid_1106'
cfgs.shelter_pickle_path = 'key_frame_pickle/shelter_data0901.pickle'
cfgs.full_im_path = '/home/gp/repos/FCN/experiment/finetune_label/s8/Image/'

