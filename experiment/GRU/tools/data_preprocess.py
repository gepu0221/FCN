import numpy as np
import pdb

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


