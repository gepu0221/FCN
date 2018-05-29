import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 448
cfg.img_h = 288
#define the grid of an image(S*S,S=32?)
cfg.grid_w = 32
cfg.grid_h = 32
cfg.multi_scale = [[512, 288]]

#the predict bounding box of each grid
cfg.n_boxes = 5
cfg.classes_name =  ["head"]
cfg.classes_num = {}
for idx, class_name in enumerate(cfg.classes_name):
    cfg.classes_num[class_name] = idx
cfg.n_classes = len(cfg.classes_name)

cfg.threshold = 0.6

cfg.weight_decay = 5e-4
cfg.unseen_scale = 0.01
cfg.unseen_epochs = 1
cfg.coord_scale = 1
cfg.object_scale = 5
cfg.class_scale = 1
cfg.noobject_scale = 1
cfg.max_box_num = 20

# cfg.anchors = [[1.07861217, 0.94927447], [1.34065023, 1.39018732], [1.86538481, 1.7361911], [1.91885292, 0.81379288], [2.43473128, 2.10715183]]
#modify from data 20180305 after cluster
cfg.anchors = [[1.00554489,0.97944079],[1.44649368,2.7858214 ],[2.20595055,1.60839398],[3.02870382,3.5573989 ],[5.00241047,5.43778195]]


# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1


cfg.max_epoch = 160


cfg.train_list = ["sku_train_20180305.txt"]
cfg.test_list = "sku_test_20180305.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True

cfg.gt_from_xml = False
cfg.gt_format = "custom"
