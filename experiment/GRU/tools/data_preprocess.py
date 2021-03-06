import numpy as np
import pdb
import tensorflow as tf

def normal_data(data):
    '''
        Transform data which has minus to [-1, 1]
    '''

    max_v = np.max(np.abs(data))
    data = data / max_v

    return data, max_v

def concat_data(flow, mask, grid):
    '''
        Concatenate flow data and mask together 
        for inpainting network input.
    '''
    _, h, w, _ = flow.shape
    mask = np.expand_dims(mask, axis=0)
    flow = flow[:, :h//grid*grid, :w//grid*grid, :]
    mask = mask[:, :h//grid*grid, :w//grid*grid, 0:2]

    input_data = np.concatenate([flow, mask], axis=2)

    mask_sum = np.sum(np.where(mask>127.5, 1, 0))
    flag = True
    if mask_sum == 0:
        flag = False

    return input_data, flag


def sdm(rect_param, gamma):
    '''
        Generate spatial discounting mask.
        param:
            rect_param: [w, h] of rect.
            gamma
    '''
    #w, h = rect_param[2], rect_param[3]
    h, w = rect_param[2], rect_param[3]

    mask = np.ones((w, h))

    for i in range(w):
        for j in range(h):
            mask[i, j] = max(
            gamma**min(i, w-i),
            gamma**min(j, h-j)
            )
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 3)

    return mask

def expand_sdm(rect_param, gamma, ed_radio):
    '''
        Generate spatial discounting mask of the inpainting area and its neigbor area.
        param:
            rect_param: [w, h] of rect.
            gamma
            ed_radio, expand_radio,the radio for inpaining area to expand.
    '''
    h, w = rect_param[2], rect_param[3]
    lx, ly = rect_param[0], rect_param[1]
    rx, ry = lx+h, ly+w
    w_ed, h_ed = w * ed_radio, h * ed_radio
    off_w, off_h = int((w_ed - w) / 2), int((h_ed - h) / 2)
    w_ed, h_ed = w + off_w*2, h + off_h*2

    mask = np.ones((w, h))
    mask_ed = np.ones((w_ed, h_ed))
    for i in range(w):
        for j in range(h):
            mask[i, j] = max(
            gamma**min(i, w-i),
            gamma**min(j, h-j)
            )

    for i in range(w_ed):
        for j in range(h_ed):
            mask_ed[i, j] = min(
            gamma**max(i, w_ed-i),
            gamma**max(j, h_ed-j)
            )
    mask_ed[off_w:off_w+w, off_h:off_h+h] = mask
    mask_ed = np.expand_dims(mask_ed, 0)
    mask_ed = np.expand_dims(mask_ed, 3)
    
    l_off0, l_off1, r_off0, r_off1 = 0, 0, 0, 0
    lx_n = lx - off_h
    ly_n = ly - off_w
    rx_n = rx + off_h
    ry_n = ry + off_w
    #print('lx: %d, ly: %d, rx: %d, ry: %d' % (lx, ly, rx, ry))
    #print('mask: ', mask.shape)
    #print('mask_ed: ', mask_ed.shape)
    #print('lx_n: %d, ly_n: %d, rx_n: %d, ry_n: %d' % (lx_n, ly_n, rx_n, ry_n))
    if lx_n < 0:
        l_off0 = np.abs(lx_n)
        lx_n = 0
    if ly_n < 0:
        l_off1 = np.abs(ly_n)
        ly_n = 0
    
    im_h, im_w = 256, 256
    if rx_n > im_h:
        r_off0 = rx_n - im_h
        rx_n = im_h
    if ry_n > im_w:
        r_off1 = ry_n - im_w
        ry_n = im_w
    #print('------------')
    #print('l_off0: %d, l_off1: %d, r_off0: %d, r_off1: %d' % (l_off0, l_off1, r_off0, r_off1))
    #print('lx_n: %d, ly_n: %d, rx_n: %d, ry_n: %d' % (lx_n, ly_n, rx_n, ry_n))

    mask_ed = mask_ed[:, l_off1:w_ed-r_off1, l_off0:h_ed-r_off0, :]
    new_rect_param = [lx_n, ly_n, rx_n-lx_n, ry_n-ly_n]    

    return mask_ed, new_rect_param


def local_patch(x, bbox):
    '''
        Crop lacal patch according to bbox.
        Args:
            x: input
            bbox: (top, left, height, weight)
        Returns:
            tf.Tensor: local patch
    '''
    #x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    i0 = bbox[0]
    j0 = bbox[1]
    i1 = i0 + bbox[2]
    j1 = j0 + bbox[3]

    #x = x[:, i0:i1, j0:j1, :]
    x = x[:, j0:j1, i0:i1, :]

    return x

def expand_local_patch(x, bbox, ed_ratio):
    '''
        Crop lacal patch according to bbox.
        Args:
            x: input
            bbox: (top, left, height, weight)
        Returns:
            tf.Tensor: local patch
    '''
    #x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    w, h = bbox[2], bbox[3]
    w_ed, h_ed = w * ed_ratio, h * ed_ratio
    off_w, off_h = int((w_ed - w) / 2), int((h_ed - h) / 2)

    i0 = bbox[0] - off_w
    j0 = bbox[1] - off_h
    i1 = i0 + bbox[2] + off_w
    j1 = j0 + bbox[3] + off_h

    x = x[:, j0:j1, i0:i1, :]

    return x
