import numpy as np
import cv2
import glob, os, pdb, random

data_dir = 'data/inst_da/s8_video_part_3000_noInst'
inst_da_dir = 'data/inst_da/inst_da'
rect_inst_da_mask_dir = 'data/inst_da/rect_inst_da_mask'
inst_da_mask_dir = 'data/inst_da/inst_da_mask'

save_dir = 'data/inst_da/rect_inst_da1215'
save_mask_dir = 'data/inst_da/rect_inst_da_mask1215'


t_idx = 5

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(save_mask_dir):
    os.makedirs(save_mask_dir)

inst_glob = os.path.join(inst_da_dir, '*.bmp')
inst_f_list = []
inst_f_list.extend(glob.glob(inst_glob))
inst_num = len(inst_f_list)

inst_list = []
inst_mask_list = []
rect_inst_mask_list = []
for f in inst_f_list:
    
    fn = os.path.splitext(f.split("/")[-1])[0]
    inst = cv2.imread(f)

    mask_path = os.path.join(inst_da_mask_dir, fn+'.bmp')
    mask = cv2.imread(mask_path)

    rect_mask_path = os.path.join(rect_inst_da_mask_dir, fn+'.bmp')
    rect_mask = cv2.imread(rect_mask_path)

    inst_list.append(inst)
    inst_mask_list.append(mask)
    rect_inst_mask_list.append(rect_mask)


file_glob = os.path.join(data_dir, '*.bmp')
file_list = []
file_list.extend(glob.glob(file_glob))

for f in file_list:
    
    fn = os.path.splitext(f.split("/")[-1])[0]

    im = cv2.imread(f)
    h, w, _ = im.shape
    inpt_mask = np.zeros((h, w, 3))
    
    choose_idx = np.random.randint(inst_num)
    inst = inst_list[choose_idx]
    mask = inst_mask_list[choose_idx]
    rect_mask = rect_inst_mask_list[choose_idx]

    i0, j0 = [np.random.randint(h),
            np.random.randint(w)]

    off = h-i0, w-j0

    i1, j1 = i0 + off[0], j0 + off[1]

    crop = inst[i0:i1, j0:j1, :]
    im_crop = im[i0:i1, j0:j1, :]
    mask = mask[i0:i1, j0:j1, :] / 255
    crop_filter = crop * mask + im_crop * (1 - mask)
    #cv2.imwrite('crop.bmp', crop_filter)

    rect_mask = rect_mask[i0:i1, j0:j1, :] / 255
    if np.sum(rect_mask) < 100:
        continue

    im[i0:i1, j0:j1, :] = crop_filter
    save_path = os.path.join(save_dir, '%s_%d.bmp' % (fn, t_idx))
    cv2.imwrite(save_path, im)

    #rect_mask = rect_mask[i0:i1, j0:j1, :] / 255
    inpt_mask[i0:i1, j0:j1, :] = rect_mask * 255
    save_path = os.path.join(save_mask_dir, '%s_%d.bmp' % (fn, t_idx))
    cv2.imwrite(save_path, inpt_mask)


    


