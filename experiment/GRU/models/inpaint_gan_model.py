import logging

import cv2, pdb
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from models.inpaint_ops import gen_conv, gen_deconv
import models.config as cfgs

logger = logging.getLogger()

class InpaintModel():
    
    def __init__(self):
        self.name = 'InpaintModel'
        # [-1, 1]?
        #self.batch_data = tf.placeholder(shape=[None, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1], cfgs.channel], dtype=tf.float32)
 

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
            #mask_s = resize_mask_like(mask, x)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gen_deconv(x, cnum, name='conv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')
            x = gen_conv(x, 2, 3, 1, activation=None, name='conv17')
            #Update by gp
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            
        return x_stage1

    def build_wgan_local_discriminator(self, x, reuse=False, training=True):

        with tf.variable_scope('discriminator_local', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*8, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')

            return dout_local




    def build_graph(self, batch_data, config, training=True, summary=False, reuse=False):
        
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        mask = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw

        batch_incomplete = batch_pos * (1. - mask)
        #inpaint
        batch_predicted = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training)
        
        # apply mask and complete image
        batch_complete = batch_predicted * mask + batch_incomplete * (1.-mask)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')

        return g_vars, batch_complete, batch_predicted
        
        
