#Generate mask for data instrument for data augumentation.
import cv2
import numpy as np
import os, glob, pdb

root_dir = 'data/inst_da'
mask_dir = 'data/inst_da_mask'

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

file_list = []
glob_file = os.path.join(root_dir, '*.bmp')
file_list.extend(glob.glob(glob_file))

for f in file_list:

    fn = os.path.splitext(f.split("/")[-1])[0]

    im = cv2.imread(f)
    im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f, im)

    sz = im.shape
    mask = np.zeros((sz[0], sz[1], 3))
    black = np.zeros((3))
    white = np.ones((3)) * 255

    for i in range(sz[0]):
        for j in range(sz[1]):
            if im[i][j][0] != 0 or im[i][j][1] != 0 or im[i][j][2] != 0:
                mask[i][j] = white



    save_path = os.path.join(mask_dir, fn+'.bmp')
    cv2.imwrite(save_path, mask)
