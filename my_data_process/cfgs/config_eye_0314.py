import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.test_ratio=0.1

cfg.train_list = ["training.txt"]
cfg.test_list = "validation.txt"

cfg.data_dir='crop'

