__author__ = 'gp'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
import cv2
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import TensorflowUtils as utils





def deconv2d_layer(x, name, W_s, output_shape=None, stride = 2):
    '''Deconv2d operator
    Args:
        x: inputs
        W_s: shape of weight
        output_shape: shape after deconv2d
    '''
    W_t = utils.weight_variable(W_s, name='W_'+name )
    b_t = utils.bias_variable([W_s[2]], name='b_'+name)
    conv_t = utils.conv2d_transpose_strided(x, W_t, b_t, output_shape, stride)
    print('conv_%s: '%name, conv_t.get_shape())

    return conv_t

#U-net
def deconv2d_layer_concat(x, name, W_s, concat_x, output_shape=None, stride=2, stddev=0.02, if_relu=False):
    '''
    Deconv2d operator for U-Net concat.
    Args:
        x: inputs
        W_s: shape of weight
        output_shape: shape after deconv2d
    '''
    if output_shape == None:
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    W_t = utils.weight_variable(W_s, stddev=stddev, name='W_'+name)
    b_t = utils.bias_variable(W_s[2], name='b_'+name)
    conv_t = utils.conv2d_transpose_strided(x, W_t, b_t, output_shape, stride)
    
    if if_relu:
        conv_t = tf.nn.relu(conv_t, name=name+'_relu')
    
    conv_concat = utils.crop_and_concat(concat_x, conv_t)

    return conv_concat
    

def conv2d_layer(x, name, W_s, pool_, if_relu=False, stride=2, stddev=0.02, keep_prob_=0):
    '''Conv2d operator
    Args:
        pool_: if pool_==0:not pooling else pooling
    '''
    W = utils.weight_variable(W_s, stddev=stddev, name='W'+name)
    b = utils.bias_variable([W_s[3]], name='b'+name)
    conv = utils.conv2d_strided(x, W, b, stride)
    print('shape after conv: ', conv.shape)
    
    if keep_prob_:
        conv = tf.nn.dropout(conv, keep_prob_)

    if if_relu:
        conv = tf.nn.relu(conv, name=name+'_relu')

    if pool_:
        conv = utils.max_pool(conv, pool_, 2)
    print('shape after pool: ', conv.shape)
    return conv

#use weight 
class U_Net_gp(object):
    
    def __init__(self, num_units_list, first_stride_list, depth_list=None, weights=None, n=0):
        '''
        Args:
            num_units_list: the number of bottleneck unit in each block
            first_stride_list: stride in branch1
            depth_list: output channles in each block ,if use weights, it is None.
            weights,n: initialize weights using mat 
        '''
        pass

    #U-net
    def down_layer_unit(self, x, fz, in_ch, out_ch, stddev=0.02, keep_prob_, name):
        
        scope_name = 'down_conv_%s' % name
        with tf.name_scope(scope_name):
            part1_name = '%s_part1' % scope_name
            with tf.variable_scope(part1_name):
                w1_shape = [fz, fz, in_ch, out_ch]
                tmp_h_conv = conv2d_layer(x, '1', w1_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=keep_prob_)

            part2_name = '%s_part2' % scope_name
            with tf.variable_scope(part2_name):
                w2_shape = [fz, fz, out_ch, out_ch]
                dw_h_conv = conv2d_layer(tep_h_conv, '2', w2_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=keep_prob_)
                
        
        return dw_h_conv
    
    #U-Net
    def up_layer_unit(self, x, concat_x, fz, pool_sz, f_ch stddev=0.02, name):
        
        scope_name = 'up_conv_%s' % name
        with tf.name_scope(scope_name):
            name = '%s_concat' % scope_name
            with tf.variable_scope(name):
                # Deconv [k_sz, k_sz, out_ch, in_ch]
                concat_w_shape = [pool_sz, pool_sz, f_ch // 2, f_ch]
                h_deconv_concat = deconv2d_layer_concat(x, name, concat_w_shape, concat_x, output_shape=None, stride=pool_sz, if_relu=True)
            
            part1_name = '%s_part1' % scope_name
            with tf.variable_scope(part1_name):
                #Number of channels is half for output.
                w1_shape = [fz, fz, f_ch, f_ch // 2]
                conv1 = conv2d_layer(h_deconv_concat, '1', w1_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=keep_prob_)
            
            part2_name = '%s_part2' % scope_name
            with tf.variable_scope(part2_name):
                #Number of channels is invariant.
                w2_shape = [fz, fz, f_ch // 2, f_ch // 2]
                conv2 = conv2d_layer(conv2, '2', w2_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=keep_prob_)
                

        return  conv2
           
            

        
    #U-net
    def u_net_op(self, x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
        '''
        Args:
            x: input data
            keep_prob: dropout probability 
            channels: number of channels of input image
            n_class: number of output labels
            layers: number of layers in the net
            features_root: number of features in the first layer
            pool_size: size of max pooling
            summaries: Flag if  summaries should be created
        '''
        #1. down layers
        dw_h_convs = {}
        for layer in range(0, layers):
            out_ch = 2** layer * features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))
            if layer == 0:
                in_ch = tf.shape(x)[3]
            else
                #// exact division
                in_ch = out_ch // 2
            name = 'down_conv_%s' % str(layer)
            x = self.down_layer_unit(x, filter_size, in_ch, out_ch, stddev, keep_prob_, str(layer))
            dw_h_convs[name] = x
                
            if layer < layers-1:
                x = utils.max_pool(x, pool_size)
        x = dw_h_convs[name]
        
        #2. up layers
        up_h_convs = {}
        for layer in range(layers-2, -1, -1):
            
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (f_sz ** 2 * features))
            concat_x = dw_h_convw['down_conv_%s' % str(layer)]
            x = self.up_layer_unit(x, concat_x, fz, pool_sz, features, stddev, name=str(layer))
            name = 'up_conv_%s' % str(layer)
            up_h_convs[name] = x
            
        #3. output map
        with tf.name_scope('output_map'):
            with tf.variable_scope('out_map'):
                w_shape = [1, 1, features_root, n_class]
                name = 'output'
                output_map = conv2d_layer(x, name=name, w_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=0)
                up_h_convs['out'] = output_map

        return output_map


          
       
