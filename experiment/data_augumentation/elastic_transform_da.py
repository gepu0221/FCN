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



def main():
    
    fn = 'img00000.bmp'
    im = cv2.imread(fn, 0)
    im = elastic_transform(im, alpha=im.shape[1]*5, sigma=im.shape[1]*0.07)
    cv2.imwrite('re'+fn, im)


if __name__=='__main__':
    main()

