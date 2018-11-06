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
    b_t = utils.weight_variable([W_s[2]], name='b_'+name)
    conv_t = utils.conv2d_transpose_strided(x, W_t, b_t, output_shape, stride)
    print('conv_%s: '%name, conv_t.get_shape())

    return conv_t


def conv2d_layer(x, name, W_s, pool_, if_relu=False, stride=2, keep_prob_=0):
    '''Conv2d operator
    Args:
        pool_: if pool_==0:not pooling else pooling
    '''
    W = utils.weight_variable(W_s, name='W'+name)
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
        self.num_units_list = num_units_list
        self.first_stride_list = first_stride_list
        self.depth_list = depth_list
        if weights is not None:
            self.weights = weights
            self.n = n
        #resnet blocks
        self.blocks = []
        self.num_blocks = len(self.num_units_list)
        self.create_blocks()

    def create_blocks(self):
        '''Create block infomation
        '''

        for i in range(self.num_blocks):
            name = 'block%d' % (i+1)
            num_units = self.num_units_list[i]
            if self.depth_list is not None:
                out_depth = self.depth_list[i]
            else:
                out_depth = 0
            first_stride = self.first_stride_list[i]
            block_info = {'name': name, 'num_units': num_units, 'out_depth': out_depth, 'first_stride': first_stride}

            self.blocks.append(block_info)
    #U-net
    def down_layer_unit(self, x, fz, in_ch, out_ch, stddev=0.02, keep_prob_, name):
        
        net = {}
        scope_name = 'down_conv_%s' % name
        with tf.name_scope(scope_name):
            part1_name = '%s_part1' % scope_name
            with tf.varibale_scope(part1_name):
                w1_shape = [fz, fz, in_ch, out_ch]
                tmp_h_conv = conv2d_layer(x, '1', w1_shape, pool_=0, if_relu=True, stride=1, keep_prob_=keep_prob_)

            part2_name = '%s_part2' % scope_name
            with tf.variable_scope(part2_name):
                w2_shape = [fz, fz, out_ch, out_ch]
                dw_h_conv = conv2d_layer(tep_h_conv, '2', w2_shape, pool_=0, if_relu=True, stride=1, keep_prob_=keep_prob_)
                
                
            

        
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

        
        return x, net


    def bn(self, tensor, name=None):
        """
        :param tensor: 4D tensor input
        :param name: name of the operation
        :return: local response normalized tensor - not using batch normalization :(
        """
        return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)
    
    def get_kernel_bias_res(self, name):
        kernels = self.weights[self.n][1]
        self.n += 2
        bias = self.weights[self.n][1]
        self.n += 2
        #out_chan = kernels.shape[i]
        kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
        #bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
        
        return kernels

    def conv_bn(self, inputs, W, stride, if_bn, if_rule, name=None):
        conv = tf.nn.conv2d(inputs, W, strides=[1, stride, stride, 1], padding='SAME', name='conv_%s' % name)
        if if_bn:
            conv = self.bn(conv, 'bn_%s' % name)
        if if_rule:
            conv = tf.nn.relu(conv, name='relu')

        return conv

    
    def bottleneck_unit(self, x, scope_name, first_stride=1, channel_equal=True):
        '''Resnet bottleneck
        Args:
            channel_equal: if need branch1
        '''
        #in_chans = x.get_shape().as_list[3]   
        #eg 2a-->res2a
        scope_name = 'res_%s' % scope_name
        print('----------------Layer: %s-------------------' % scope_name)
        print('input:',x.get_shape())
        with tf.variable_scope(scope_name):
            if channel_equal:
                b1 = x
            else:
                with tf.variable_scope('branch1'):
                    name_ = '%s_branch1' % scope_name
                    branch1_w = self.get_kernel_bias_res(name = name_)
                    b1 = self.conv_bn(x, branch1_w, stride=first_stride, if_bn=True, if_rule=False, name=name_)
                    print(name_, branch1_w.get_shape())
            with tf.variable_scope('branch2a'):
                name_ = '%s_branch2a' % scope_name
                branch2a_w = self.get_kernel_bias_res(name= name_)
                print(name_, branch2a_w.get_shape())
                #conv
                b2 = self.conv_bn(x, branch2a_w, stride=first_stride, if_bn=True, if_rule=True, name=name_)    

            with tf.variable_scope('branch2b'):
                name_ = '%s_branch2b' % scope_name
                branch2b_w = self.get_kernel_bias_res(name=name_)
                print(name_, branch2a_w.get_shape())
                #conv
                b2 = self.conv_bn(b2, branch2b_w, stride=1, if_bn=True, if_rule=True, name=name_)    

            with tf.variable_scope('branch2c'):
                name_ = '%s_branch2c' % scope_name
                branch2c_w = self.get_kernel_bias_res(name=name_)
                print(name_, branch2a_w.get_shape())
                #conv
                b2 = self.conv_bn(b2, branch2c_w, stride=1, if_bn=True, if_rule=False, name=name_)

            x = b1 + b2
            print('output: ', x.get_shape())
            return tf.nn.relu(x, name='relu')





           
       
