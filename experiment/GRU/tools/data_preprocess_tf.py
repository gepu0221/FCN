import numpy as np
import pdb
import tensorflow as tf

def normal_data(data):
    '''
        Transform data which has minus to [-1, 1]
    '''

    max_v = tf.reduce_max(tf.abs(data))
    data = data / max_v

    return data, max_v

def concat_data(flow, mask):
    '''
        Concatenate flow data and mask together 
        for inpainting network input.
    '''
    
    mask = mask[:, :, :, 0:2]
    input_data = tf.concat([flow, mask], axis=2)


    return input_data


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
    w, h = rect_param[2], rect_param[3]
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
            mask[i, j] = min(
            gamma**max(i, w_ed-i),
            gamma**max(j, h_ed-j)
            )
    mask_ed[off_w:off_w+w, off_h:off_h+h] = mask
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 3)

    return mask


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

def expand_local_patch(x, bbox):
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
    w_ed, h_ed = w * ed_radio, h * ed_radio
    off_w, off_h = int((w_ed - w) / 2), int((h_ed - h) / 2)

    i0 = bbox[0] - off_w
    j0 = bbox[1] - off_h
    i1 = i0 + bbox[2] + off_w
    j1 = j0 + bbox[3] + off_h

    x = x[:, i0:i1, j0:j1, :]

    return x
