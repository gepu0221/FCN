#Data augmentation.
import tensorflow as tf
from BatchReader_multi_ellip import *
import cv2

try:
    from .cfgs.config_train_m import cfgs 
except Exception:
    from cfgs.config_train_m import cfgs


def translate_images(im, translations):
    pass

def random_crop(im, anno, crop_w, crop_h, padding_w, padding_h, ignore_label = 255):
    '''
    Random crop
    [crop_w, crop_h]: random crop size
    [padding_w, padding_h]: size padding image to crop including padding part.
    Defalut channel of image and annotation is 1.
    '''
    anno = anno - ignore_label #latter add due to 0 padding.

    #combine image and annotation together and padding if case out of range.
    data_pre = tf.concat((im, anno), 2)
    im_shape = tf.shape(im)
    data_pre = tf.image.pad_to_bounding_box(data_pre, 0, 0, tf.maximum(padding_h, im_shape[0]), tf.maximum(padding_w, im_shape[1]))
    
    im_ch = tf.shape(im)[-1] #channle of image
    anno_ch = tf.shape(anno)[-1]
    data_c = tf.random_crop(data_pre, [crop_h, crop_w, tf.shape(data_pre)[-1]])
    
    im_c = data_c[:,:, :im_ch]
    anno_c = data_c[:,:, im_ch:] + ignore_label
    anno_c = tf.cast(anno_c, dtype=tf.int32)

    #Set static shape so that tensorflow knows shape at complie time.
    im_c.set_shape((crop_h, crop_w, cfgs.seq_num+1))
    anno_c.set_shape((crop_h, crop_w, 1))

    #unstack random crop result.
    #im_c, anno_c = tf.unstack(data_c, axis=2) 

    return im_c, anno_c

def random_crop_batch(im, anno, crop_w, crop_h, padding_w, padding_h, ignore_label = 255):
    '''
    Random crop on image batches
    [crop_w, crop_h]: random crop size
    [padding_w, padding_h]: size padding image to crop including padding part.
    Defalut channel of image and annotation is 1.
    '''
    anno = anno - ignore_label #latter add due to 0 padding.

    #combine image and annotation together and padding if case out of range.
    data_pre = tf.concat((im, anno), 3)
    im_shape = tf.shape(im)
    data_pre = tf.image.pad_to_bounding_box(data_pre, 0, 0, tf.maximum(padding_h, im_shape[1]), tf.maximum(padding_w, im_shape[2]))
    
    im_ch = tf.shape(im)[-1] #channle of image
    anno_ch = tf.shape(anno)[-1]
    data_c = tf.random_crop(data_pre, [tf.shape(data_pre)[0], crop_h, crop_w, tf.shape(data_pre)[-1]])
    
    im_c = data_c[:,:,:, :im_ch]
    anno_c = data_c[:,:,:, im_ch:] + ignore_label
    anno_c = tf.cast(anno_c, dtype=tf.int32)

    #Set static shape so that tensorflow knows shape at complie time.
    im_c.set_shape((cfgs.batch_size, crop_h, crop_w, cfgs.seq_num+1))
    anno_c.set_shape((cfgs.batch_size, crop_h, crop_w, 1))



    #unstack random crop result.
    #im_c, anno_c = tf.unstack(data_c, axis=2) 

    return im_c, anno_c



if __name__ == '__main__':
    im_name = tf.placeholder(dtype=tf.string, name='im_name')
    anno_name = tf.placeholder(dtype=tf.string, name='anno_name')
    
    im = tf.cast(transform_gray(im_name), tf.int32)
    anno = tf.cast(transform_anno_test(anno_name), tf.int32)

    im_c, anno_c = random_crop(im, anno, 112, 112, 300, 300)
    im_n ="im.bmp"
    anno_n="anno.bmp"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        im_, anno_ = sess.run([im_c, anno_c], feed_dict={im_name: im_n, anno_name: anno_n})
        cv2.imwrite('im_r.bmp', im_)
        cv2.imwrite('anno_r.bmp', anno_)
        sz_ = im_.shape
        for i in range(sz_[0]):
            for j in range(sz_[1]):
                if anno_[i][j][0] == 1:
                    im_[i][j][0] = 1
                elif anno_[i][j][0] != 0:
                    print('(%d, %d) %d' % (i, j, im_[i][j][0]))
       
        cv2.imwrite('label_r.bmp', im_)

