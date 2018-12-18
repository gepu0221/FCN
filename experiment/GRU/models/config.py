import os

data_path = '/home/gp/repos/FCN/experiment/GRU/data/flow_dilate_pickle'
mask_path = '/home/gp/repos/FCN/experiment/GRU/view/s8_gru_flow_mask_dilate1210/train'
pickle_save_path = '/home/gp/repos/FCN/experiment/GRU/data/inpt_flow_dilate_pickle'
checkpoint_dir = 'model_logs/Places2'

batch_size = 1
#IMAGE_SIZE = [536, 1920]
IMAGE_SIZE = [256, 512]
channel = 3

