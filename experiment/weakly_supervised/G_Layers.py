__author__ = 'gp'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
import cv2
import pdb
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import TensorflowUtils as utils

try:
    from .cfgs.config_train_u_net import cfgs
except Exception:
    from cfgs.config_train_u_net import cfgs





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
    b_t = utils.bias_variable([W_s[2]], name='b_'+name)
    #conv_t = utils.conv2d_transpose_strided_valid(x, W_t, b_t, output_shape, stride)
    conv_t = utils.conv2d_transpose_strided(x, W_t, b_t, output_shape, stride)
    
   
    if if_relu:
        conv_t = tf.nn.relu(conv_t, name=name+'_relu')
    
    conv_concat = utils.crop_and_concat(concat_x, conv_t)

    return conv_concat
    

def conv2d_layer(x, name, W_s, pool_, if_relu=False, stride=2, stddev=0.02, if_dropout=False, keep_prob_=1):
    '''Conv2d operator
    Args:
        pool_: if pool_==0:not pooling else pooling
    '''
    W = utils.weight_variable(W_s, stddev=stddev, name='W'+name)
    b = utils.bias_variable([W_s[3]], name='b'+name)
    #conv = utils.conv2d_strided_valid(x, W, b, stride)
    conv = utils.conv2d_strided(x, W, b, stride)

    print('shape after conv: ', conv.shape)
    print('--------------------------------')
    
    if if_dropout:
        conv = tf.nn.dropout(conv, keep_prob_)

    if if_relu:
        conv = tf.nn.relu(conv, name=name+'_relu')

    if pool_:
        conv = utils.max_pool(conv, pool_, 2)
        print('shape after pool: ', conv.shape)
    return conv

#use weight 
class U_Net_gp(object):
    
    def __init__(self):
        '''
        Args:
            num_units_list: the number of bottleneck unit in each block
            first_stride_list: stride in branch1
            depth_list: output channles in each block ,if use weights, it is None.
            weights,n: initialize weights using mat 
        '''
        pass

    #U-net
    def down_layer_unit(self, x, fz, in_ch, out_ch, stddev=0.02, keep_prob_=1, if_dropout=False, name=None):
        
        scope_name = 'down_conv_%s' % name
        with tf.name_scope(scope_name):
            part1_name = '%s_part1' % scope_name
            with tf.variable_scope(part1_name):
                w1_shape = [fz, fz, in_ch, out_ch]
                tmp_h_conv = conv2d_layer(x, '1', w1_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=keep_prob_)

            part2_name = '%s_part2' % scope_name
            with tf.variable_scope(part2_name):
                w2_shape = [fz, fz, out_ch, out_ch]
                dw_h_conv = conv2d_layer(tmp_h_conv, '2', w2_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, keep_prob_=keep_prob_)
                
        
        return dw_h_conv
    
    #U-Net
    def up_layer_unit(self, x, concat_x, fz, pool_sz, f_ch, stddev=0.02, keep_prob_=1, if_dropout=False, name=None):
        
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
                conv2 = conv2d_layer(conv1, '2', w2_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, if_dropout=if_dropout, keep_prob_=keep_prob_)
                

        return  conv2
           
            

        
    #U-net
    def u_net_op(self, x, keep_prob_, channels, n_class, layers=3, class_convs_num=1, features_root=16, filter_size=3, pool_size=2, summaries=True):
        '''
        Args:
            x: input data
            keep_prob: dropout probability 
            channels: number of channels of input image
            n_class: number of output labels
            layers: number of layers in the u-net
            class_convs_num: number of conv operator after u-net down layers operator.
            features_root: number of features in the first layer
            pool_size: size of max pooling
            summaries: Flag if  summaries should be created
        '''
        #1. down layers
        dw_h_convs = {}
        for layer in range(0, layers):
            out_ch = 2** layer * features_root
            stddev = np.sqrt(2 / (filter_size**2 * out_ch))
            if layer == 0:
                in_ch = channels
            else:
                #// exact division
                in_ch = out_ch // 2
            name = 'down_conv_%s' % str(layer)
            x = self.down_layer_unit(x, filter_size, in_ch, out_ch, stddev, keep_prob_, if_dropout=True, name=str(layer))
            
            dw_h_convs[name] = x
                
            if layer < layers-1:
                x = utils.max_pool_valid(x, kernel_size=pool_size, stride=pool_size)
        x = dw_h_convs[name]
        
        #2. label occlusion
        print('--------label occlusion-------------')
        x_class = x
        ch_class = out_ch
        for i in range(class_convs_num):
            print('class conv %d' % i)
            scope_name= 'class_conv_%s' % str(i)
            with tf.name_scope(scope_name):
                var_name = 'class_conv_var_%s' % str(i)
                with tf.variable_scope(var_name):
                    w_class_s = [filter_size, filter_size, ch_class, ch_class]
                    x_class = conv2d_layer(x_class, str(i), w_class_s, pool_=2, if_relu=True, stride=2)
        
        sz_class = tf.shape(x_class)
        
        p_k_sz = cfgs.p_k_sz
        x_class = utils.avg_pool_diff(x_class, p_k_sz[0], p_k_sz[1], stride=1)
        with tf.name_scope('full_conn'):
            with tf.variable_scope('full_conn_var'):
                fc_w = utils.weight_variable([1, 1, ch_class, 2], name="fc_w")
                fc_b = utils.bias_variable([2], name="fc_b")
                x_class = utils.conv2d_basic(x_class, fc_w, fc_b)
                class_logits = tf.squeeze(x_class, [1,2])
        print('--------label occlusion end----------')

        #3. up layers
        up_h_convs = {}
        for layer in range(layers-2, -1, -1):
                
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            concat_x = dw_h_convs['down_conv_%s' % str(layer)]
            x = self.up_layer_unit(x, concat_x, filter_size, pool_size, features, stddev, keep_prob_, if_dropout=True, name=str(layer))
            name = 'up_conv_%s' % str(layer)

            #if cfgs.if_pad[layer]:
                #paddings = tf.constant([[0,0], [0,0], [1,0], [0,0]])
                #x = tf.pad(x, paddings, 'CONSTANT')
            up_h_convs[name] = x
            
        #4. output map
        with tf.name_scope('output_map'):
            with tf.variable_scope('out_map'):
                w_shape = [1, 1, features_root, n_class]
                name = 'output'
                output_map = conv2d_layer(x, name, w_shape, pool_=0, if_relu=True, stride=1, stddev=stddev, if_dropout=True, keep_prob_=keep_prob_)
                up_h_convs['out'] = output_map

        return output_map, class_logits


          
       
