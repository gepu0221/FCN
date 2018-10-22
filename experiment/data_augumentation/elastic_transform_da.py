from scipy.ndimage.filters import gaussian_filter
import time
from scipy.ndimage.interpolation import map_coordinates
import cv2
import numpy as np

def elastic_transform(x, alpha, sigma, mode='constant', cval=0, is_random=False):
    '''
    Elastic transformation for image as described in simard2003
    Args:
        x: a grayscale image.
        alpha: float, value for elastic transformation.
        sigma: float or sequence of float
               Standard deviation for Gaussian kernel. The smaller the sigma, the more transformation.
        mode: str , mode of guassian filter
        cval: the value outside the image boundaries.
        is_random: default is False.
    '''
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))

    is_3d = False
    #if len(x.shape) == 3 and x.shape[-1] == 1:
        #x = x[:, :, 0]
        #is_3d = True
    #else:
        #raise AssertionError('Input should be grayscale image')

    is_3d = True
    shape = x.shape
    



    #*shape: pass the element of shape one by one
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha

    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
    if is_3d:
        #Map the input array to new coordinates by interpolation.
        return map_coordinates(x, indices, order=1).reshape((shape[0], shape[1], 1)) 
    else:
        return map_corrdinates(x, indices, order=1).reshape(shape)

# Get offset of elastic transform
def elastic_transform_offset(shape, alpha, sigma, mode='constant', cval=0, is_random=False):
    '''
    Elastic transformation for image as described in simard2003
    Args:
        shape: the shape of a grayscale image.
        alpha: float, value for elastic transformation.
        sigma: float or sequence of float
               Standard deviation for Gaussian kernel. The smaller the sigma, the more transformation.
        mode: str , mode of guassian filter
        cval: the value outside the image boundaries.
        is_random: default is False.
    '''
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
    

    #*shape: pass the element of shape one by one
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha

    return dx, dy


def elastic_transform_im_anno(im, anno, alpha, sigma, mode='constant', cval=0, is_random=False):

    shape = im.shape
    dx, dy = elastic_transform_offset(shape, alpha, sigma, mode=mode, cval=cval)

    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))


    #Map the input array to new coordinates by interpolation.
    im_e = map_coordinates(im, indices, order=1).reshape((shape[0], shape[1], 1)) 
    anno_e = map_coordinates(anno, indices, order=1).reshape((shape[0], shape[1], 1)) 

    
    return im_e, anno_e


def main():
    
    fn = 'img00000.bmp'
    anno_fn = 's8img00000.bmp'
    im = cv2.imread(fn, 0)
    rate = 2
    sz = im.shape
    im = cv2.resize(im, (int(sz[1]/rate), int(sz[0]/rate)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('re_ori_'+fn, im)
    anno = cv2.imread(anno_fn, 0)
    #im = elastic_transform(im, alpha=im.shape[1]*5, sigma=im.shape[1]*0.07)
    im_e, anno_e = elastic_transform_im_anno(im, anno, alpha=im.shape[1]*2, sigma=im.shape[1]*0.06)
    
    cv2.imwrite('re_006'+fn, im_e)
    cv2.imwrite('re_006'+anno_fn, anno_e)
    
    sz = im_e.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            if anno_e[i][j] == 255:
                im_e[i][j] = 255
    
    cv2.imwrite('re_show_006'+fn, im_e)

if __name__=='__main__':
    main()

