import numpy as np
import tensorflow as tf
import pdb

try:
    from .cfgs.config import cfgs
except Exception:
    from cfgs.config import cfgs


class STGRU:
    def __init__(self, tensor_size, conv_sizes, bilinear_warping_module):
        # tensor_size is something like 19 x 512 x 512
        # conv sizes are e.g. 5 x 5
        self.bilinear_warping_module = bilinear_warping_module
        channels, height, width = tensor_size
        conv_height, conv_width = conv_sizes
        conv_pad = conv_height / 2

        self.channels, self.height, self.width = channels, height, width
        self.conv_height, self.conv_width = conv_height, conv_width

        identity = np.zeros((conv_height, conv_width, channels, channels))
        for k in range(channels):
          identity[conv_height//2, conv_width//2, k, k] = 1.
        identity_map = tf.constant(identity, dtype=tf.float32)
        # identity + noise was needed for some variables to train the model 
        self.weights = {
            'ir': tf.Variable(tf.random_normal([conv_height, conv_width, 3, 1], stddev=0.001), name="W_ir"),
            'xh': tf.Variable(6.*identity_map + tf.random_normal([conv_height, conv_width, channels, channels], stddev=0.01), name="W_xh"),
            'hh': tf.Variable(6.*identity_map + tf.random_normal([conv_height, conv_width, channels, channels], stddev=0.01), name="W_hh"),
            'xz': tf.Variable(tf.random_normal([conv_height, conv_width, channels, 1], stddev=0.01), name="W_xz"),
            'hz': tf.Variable(tf.random_normal([conv_height, conv_width, channels, 1], stddev=0.01), name="W_hz"),
            'lambda': tf.Variable(tf.constant(2., dtype=tf.float32), name="lambda"),
            'bias_r': tf.Variable(tf.zeros([1], dtype=tf.float32), name="bias_r"),
            'bias_z': tf.Variable(tf.zeros([channels], dtype=tf.float32), name="bias_z"),
        }

    def get_one_step_predictor(self):
        input_images_tensor = tf.placeholder('float', [2, 1, self.height, self.width, 3], name="gru_input_images")
        input_images = tf.unstack(input_images_tensor, num=2)

        input_flow = tf.placeholder('float', [1, self.height, self.width, 2], name="gru_input_flows")
        
        input_segmentation = tf.placeholder('float', [1, self.height, self.width, self.channels], name="gru_input_unaries")
        
        prev_h = tf.placeholder('float', [1, self.height, self.width, self.channels])
        
        new_h = self.get_GRU_cell(input_images[1], input_images[0], \
             input_flow, prev_h, input_segmentation)

        prediction = tf.argmax(new_h, 3)
        return input_images_tensor, input_flow, input_segmentation, prev_h, new_h, prediction

    def get_optimizer(self, N_steps):
        input_images_tensor = tf.placeholder('float', [N_steps, 1, self.height, self.width, 3], name="gru_input_images")
        input_images = tf.unstack(input_images_tensor, num=N_steps)

        input_flow_tensor = tf.placeholder('float', [N_steps-1, 1, self.height, self.width, 2], name="gru_input_flows")
        input_flow = tf.unstack(input_flow_tensor, num=N_steps-1)

        input_segmentation_tensor = tf.placeholder('float', [N_steps, 1, self.height, self.width, self.channels], name="gru_input_unaries")
        input_segmentation = tf.unstack(input_segmentation_tensor, num=N_steps)

        outputs = [input_segmentation[0]]
        for t in range(1, N_steps):
            h = self.get_GRU_cell(input_images[t], input_images[t-1], \
                input_flow[t-1], outputs[-1], input_segmentation[t])
            outputs.append(h)

        # the loss is tricky to implement since softmaxloss requires [i,j] matrix
        # with j ranging over the classes
        # the image has to be manipulated to fit
        scores = tf.reshape(outputs[-1], [self.height*self.width, self.channels])
        prediction = tf.argmax(scores, 1)
        prediction = tf.reshape(prediction, [self.height, self.width])

        targets = tf.placeholder('int64', [self.height, self.width])
        targets_r = tf.reshape(targets, [self.height*self.width])
        idx = targets_r < self.channels # classes are 0,1,...,c-1 with 255 being unknown
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(scores, idx), labels=tf.boolean_mask(targets_r, idx)))
        
        #Add by gp
        pred_pro = tf.reshape(tf.nn.softmax(tf.boolean_mask(scores, idx)),[self.height, self.width, self.channels])

        learning_rate = tf.placeholder('float', [])
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95, beta2=0.99, epsilon=1e-8)
        
        opt = opt.minimize(loss)
        return opt, loss, prediction, pred_pro, learning_rate, \
          input_images_tensor, input_flow_tensor, input_segmentation_tensor, targets
    
    #Add by gp on 2018/12/05
    def slow_change_loss(self, h_prev, h, h_static):

        '''
            Caculate slow change loss.
            Args:
                h_prev: logits of prev frame.
                h: logits of current frame.
                h_static: logits of current frame through static frame net(U_net).
        '''
        
        sz = [cfgs.batch_size, cfgs.IMAGE_SIZE[0], cfgs.IMAGE_SIZE[1]]
        im_comp = tf.ones(sz, dtype=tf.int64)

        h_pro = tf.nn.softmax(h)
        h_prev_pro = tf.nn.softmax(h_prev)
        h_static_pro = tf.nn.softmax(h_static)

        #Generate instrument mask using cfgs.inst_low_pro using current h_stctic_pro
        #cur_static_inst_mask = tf.expand_dims(tf.where(tf.less_equal(h_static_pro[:, :, :, 1], cfgs.inst_low_pro), 1-im_comp, im_comp), dim=3)

        #Generate corean mask using cfgs.low_pro
        #cur_corn_mask = tf.expand_dims(tf.where(tf.less_equal(h_pro[:, :, :, 2], cfgs.low_pro), 1-im_comp, im_comp), dim=3)
        #prev_corn_mask = tf.expand_dims(tf.where(tf.less_equal(h_prev_pro[:, :, :, 2], cfgs.low_pro), 1-im_comp, im_comp), dim=3)
        
        cur_static_inst_mask = h_static_pro[:, :, :, 1]
        cur_corn_mask = h_pro[:, :, :, 2]
        prev_corn_mask = h_prev_pro[:, :, :, 2]

        cur_filter = tf.multiply(cur_static_inst_mask, cur_corn_mask)
        prev_filter = tf.multiply(cur_static_inst_mask, prev_corn_mask)

        loss = tf.reduce_mean(tf.pow((cur_filter - prev_filter), 2))

        return loss


    def get_optimizer_slow_change(self, N_steps):
        '''
            Slow change: prev frame and current frame shouldn't change abruptly. 
        '''
        input_images_tensor = tf.placeholder('float', [N_steps, 1, self.height, self.width, 3], name="gru_input_images")
        input_images = tf.unstack(input_images_tensor, num=N_steps)

        input_flow_tensor = tf.placeholder('float', [N_steps-1, 1, self.height, self.width, 2], name="gru_input_flows")
        input_flow = tf.unstack(input_flow_tensor, num=N_steps-1)

        input_segmentation_tensor = tf.placeholder('float', [N_steps, 1, self.height, self.width, self.channels], name="gru_input_unaries")
        input_segmentation = tf.unstack(input_segmentation_tensor, num=N_steps)

        outputs = [input_segmentation[0]]
        loss_slow_ch = 0
        for t in range(1, N_steps):
            h, h_prev_warped = self.get_GRU_cell(input_images[t], input_images[t-1], \
                input_flow[t-1], outputs[-1], input_segmentation[t])
            outputs.append(h)
            loss_slow_ch += self.slow_change_loss(h_prev_warped, h, input_segmentation[t])

        # the loss is tricky to implement since softmaxloss requires [i,j] matrix
        # with j ranging over the classes
        # the image has to be manipulated to fit
        scores = tf.reshape(outputs[-1], [self.height*self.width, self.channels])
        prediction = tf.argmax(scores, 1)
        prediction = tf.reshape(prediction, [self.height, self.width])

        targets = tf.placeholder('int64', [self.height, self.width])
        targets_r = tf.reshape(targets, [self.height*self.width])
        idx = targets_r < self.channels # classes are 0,1,...,c-1 with 255 being unknown
        loss_gt = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(scores, idx), labels=tf.boolean_mask(targets_r, idx)))
        #pdb.set_trace()
        #loss = tf.add(tf.cast(loss_slow_ch, tf.float32), loss_gt)
        loss = tf.cast(loss_slow_ch, tf.float32) + loss_gt
        #Add by gp
        pred_pro = tf.reshape(tf.nn.softmax(tf.boolean_mask(scores, idx)),[self.height, self.width, self.channels])

        learning_rate = tf.placeholder('float', [])
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95, beta2=0.99, epsilon=1e-8)
        
        opt = opt.minimize(loss)
        return opt, loss, prediction, pred_pro, learning_rate, \
          input_images_tensor, input_flow_tensor, input_segmentation_tensor, targets

    def get_GRU_cell(self, input_image, prev_image, flow_input, h_prev, unary_input):
        # apply softmax to h_prev and unary_input
        h_prev = self.softmax_last_dim(h_prev)
        unary_input = self.softmax_last_dim(unary_input)
        h_prev = h_prev - 1./19
        unary_input = unary_input - 1./19

        I_diff = input_image - self.bilinear_warping_module.bilinear_warping(prev_image, flow_input)
        self.prev_warping = self.bilinear_warping_module.bilinear_warping(prev_image, flow_input)
        # candidate state
        h_prev_warped = self.bilinear_warping_module.bilinear_warping(h_prev, flow_input)

        r = 1. - tf.tanh(tf.abs(tf.nn.conv2d(I_diff, self.weights['ir'], [1,1,1,1], padding='SAME') \
            + self.weights['bias_r']))
        
        h_prev_reset = h_prev_warped * r

        h_tilde = tf.nn.conv2d(unary_input, self.weights['xh'], [1,1,1,1], padding='SAME') \
          + tf.nn.conv2d(h_prev_reset, self.weights['hh'], [1,1,1,1], padding='SAME')

        
        # weighting
        z = tf.sigmoid( \
            tf.nn.conv2d(unary_input, self.weights['xz'], [1,1,1,1], padding='SAME') \
            + tf.nn.conv2d(h_prev_reset, self.weights['hz'], [1,1,1,1], padding='SAME') \
            + self.weights['bias_z']
          )

        h = self.weights['lambda']*(1 - z)*h_prev_reset + z*h_tilde

        return h, h_prev_warped

    def softmax_last_dim(self, x):
        # apply softmax to a 4D tensor along the last dimension
        S = tf.shape(x)
        y = tf.reshape(x, [-1, S[4-1]])
        y = tf.nn.softmax(y)
        y = tf.reshape(y, S)
        return y
