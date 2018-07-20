#This is FCN for sequence
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import math
import os
import TensorflowUtils as utils
import read_data as scene_parsing
import datetime
import pdb
#import BatchReader_multi as dataset
from BatchReader_multi import get_data_
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

try:
    from .cfgs.config_train_m import cfgs
except Exception:
    from cfgs.config_train_m import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE

class FCNNet(object):

    def __init__(self, mode, max_epochs, batch_size, n_classes, train_records, valid_records, im_sz, init_lr, keep_prob):
        self.max_epochs = max_epochs
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.NUM_OF_CLASSESS = n_classes
        self.IMAGE_SIZE = im_sz
        self.graph = tf.get_default_graph()
        self.learning_rate = tf.placeholder(dtype=tf.int32, name='learning_rate')
        self.anno_accu = tf.placeholder(tf.int32, shape=[None, self.IMAGE_SIZE, self.IMAGE_SIZE, 1], name='accu_anno')
        self.learning_rate = init_lr
        self.mode = mode
        self.logs_dir = cfgs.logs_dir
        self.current_itr_var = tf.Variable(0, dtype=tf.int32, name='current_itr', trainable=True)
        self.cur_epoch = tf.Variable(1, dtype=tf.int32, name='cur_epoch', trainable=False)

        vis = True if self.mode == 'all_visualize' else False
        if_anno = True if self.mode == 'train' else False
        self.image_options = {'resize': True, 'resize_size':self.IMAGE_SIZE, 'visualize':vis, 'annotation': if_anno}
        self.train_records = train_records
        self.valid_records = valid_records

        

    def get_data(self):
        with tf.device('/cpu:0'):
            #train_dataset_reader = dataset.BatchDatset(self.train_records, self.image_options)
            #valid_dataset_reader = dataset.BatchDatset(self.valid_records, self.image_options)

            #self.images, self.annotations, _, self.train_init = get_data_from_filelist(self.train_records, self.batch_size, 'get_data_train')
            #self.images, self.annotations, _, self.valid_init = get_data_from_filelist(self.valid_records, self.batch_size, 'get_data_valid')
            self.images, self.annotations, _, self.train_init, self.valid_init = get_data_(self.train_records, self.valid_records, self.batch_size)


    def vgg_net(self, weights, image):
        layers = (
            #'conv1_1', 'relu1_1',
            'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i+2][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = utils.avg_pool_2x2(current)
            net[name] = current

        return net


    def inference(self, keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up vgg initialized conv layers ...")
        model_data = utils.get_model_data(cfgs.model_dir, MODEL_URL)

        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean)
        self.mean_ = mean_pixel
        weights = np.squeeze(model_data['layers'])

        processed_image = utils.process_image(self.images, mean_pixel)

        with tf.variable_scope("inference"):
            W1 = utils.weight_variable([3, 3, 7, 64], name="W1")
            b1 = utils.bias_variable([64], name="b1")
            #conv1 = utils.conv2d_basic(self.images, W1, b1)
            conv1 = utils.conv2d_basic(processed_image, W1, b1)
            relu1 = tf.nn.relu(conv1, name='relu1')
            
            #pretrain
            image_net = self.vgg_net(weights, relu1)
            
            conv_final_layer = image_net["conv5_3"]

            pool5 = utils.max_pool_2x2(conv_final_layer)

            W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
            b6 = utils.bias_variable([4096], name="b6")
            conv6 = utils.conv2d_basic(pool5, W6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
            b7 = utils.bias_variable([4096], name="b7")
            conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            '''
            if cfgs.debug:
                utils.add_activation_summary(relu7)'''
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
            b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
            conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            deconv_shape1 = image_net["pool4"].get_shape()
            W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
            b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
            conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

            deconv_shape2 = image_net["pool3"].get_shape()
            W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
            conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

            shape = tf.shape(self.images)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
            W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
            b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
            conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        self.pred_annotation = tf.expand_dims(annotation_pred, dim=3)
        self.logits = conv_t3


    def train_optimizer(self):
        var_list = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        '''
        grads = optimizer.compute_gradients(self.loss, var_list=var_list)
        if cfgs.debug:
            for grad, var in grads:
                utils.add_gradient_summary(grad, var)
        self.train_op = optimizer.apply_gradients(grads)'''

    def loss(self):
        self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                          name="entropy")))

        self.valid_loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.squeeze(self.annotations, squeeze_dims=[3]),
                                                                          name="valid_entropy")))
    def calculate_acc(self):
        with tf.name_scope('accu'):
            pass
    
    
    def summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('train_loss', self.loss)
            tf.summary.scalar('valid_loss', self.valid_loss)
            #tf.summary.scalar('acc', self.accu_pixel)
            #tf.summary.scalar('iou_acc', self.accu_iou)
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        #bulid the graph
        self.get_data()
        self.inference(self.keep_prob)
        self.loss()
        self.train_optimizer()
        #self.calculate_acc()
        self.summary()

    def try_update_lr(self):
        try:
            with open(cfgs.learning_rate_path) as f:
                lr_ = float(f.readline().split('\n')[0])
                if self.learning_rate != lr_:
                    self.learning_rate = lr_
                    print('learning rate change from to %g' % self.learning_rate)
        except:
            pass

    def recover_model(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restore finished')
        
        return saver

    #Evaluate all validation dataset once 
    
    def valid_once(self, sess, writer, epoch, step):
        sess.run(self.valid_init)

        count = 0
        sum_acc = 0
        sum_acc_iou = 0
        acc =1
        acc_iou=1
        t0 = time.time()

        try:
            total_loss = 0
            while True:
                count +=1
                #summary_str, loss, acc, acc_iou= sess.run(fetches=[self.summary_op, self.valid_loss, self.accu, self.accu_iou], feed_dict={self.anno_accu: self.valid_annotations, self.images:images_, self.annotations:anno})
                summary_str, loss= sess.run(fetches=[self.summary_op, self.valid_loss])


                writer.add_summary(summary_str, global_step=step)
                sum_acc += acc
                sum_acc_iou += acc_iou
                total_loss += loss
                print('\r' + 32 * ' ', end='')
                print('epoch %5d\t learning_rate = %g\t step = %4d\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, step, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))


        except tf.errors.OutOfRangeError:
            count -= 1
            print('epoch %5d\t learning_rate = %g\t loss = %.3f\t valid_accuracy = %.2f%%\t valid_iou_accuracy = %.2f%%' % 
            (epoch, self.learning_rate, total_loss/count, sum_acc/count, sum_acc_iou/count))
            print('Take time %3.1f' % (time.time() - t0))
    

    
    
    def train_one_epoch(self, sess, writer, epoch, step):
        sess.run(self.train_init)
        sum_acc = 0
        sum_acc_iou = 0
        count = 0
        total_loss = 0
        t0 =time.time()
        acc = 1
        acc_iou =1
        try:
            while True:
                step += 1
                count += 1
                _, summary_str, loss, anno_, images_ = sess.run(
                fetches=[self.train_op, self.summary_op, self.loss, self.annotations, self.images])
                if math.isnan(loss) or math.isinf(loss):
                    print('loss is ', loss)
                    img = images[0]
                    img = img + self.mean_
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join('image', 'loss ' + str(loss) + ' ' + str(step) + '.jpg'), img)
                    continue

                sum_acc += acc
                sum_acc_iou += acc_iou
                total_loss += loss
                time_consumed = time.time() - t0
                time_pre_batch = time_consumed/count
                print('\r' + 32 * ' ', end='')
                print('epoch %5d\t learning_rate = %g\t step = %4d\t loss = %.3f\t mean_loss=%.3f\t train_accuracy = %.2f%%\t train_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, step, loss, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))



        except tf.errors.OutOfRangeError:
            count -= 1
            print('epoch %5d\t learning_rate = %g\t mean_loss = %.3f\t train_accuracy = %.2f%%\t train_iou_accuracy = %.2f%%' % (epoch, self.learning_rate, (total_loss/count), (sum_acc/count), (sum_acc_iou/count)))
            print('Take time %3.1f' % (time.time() - t0))
     
        return step

    def train(self):
        #Check if has the log file
        if not os.path.exists(self.logs_dir):
            print("The logs path '%s' is not found" % self.logs_dir)
            print("Create now..")
            os.makedirs(self.logs_dir)
            print("%s is created successfully!" % self.logs_dir)

        print('prepare to train...')
        writer = tf.summary.FileWriter(logdir=self.logs_dir, graph=self.graph)

        print('The graph path is %s' % self.logs_dir)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        #config = tf.ConfigProto(log_device_placement=False, allow_soft_palcement=True)
        with tf.device('/gpu:0'):
            with tf.Session(config=config) as sess:
                #1. initialize all variables
                sess.run(tf.global_variables_initializer())

                #2. Try to recover model
                saver = self.recover_model(sess)
                step = self.current_itr_var.eval()
                cur_epoch = self.cur_epoch.eval()
                print(self.current_itr_var.eval())

                #3. start to train 
                for epoch in range(cur_epoch, self.max_epochs + 1):
                    
                    #3.1 try to change learning rate
                    self.try_update_lr()
                    if epoch != 0 and epoch % 20 == 0:
                        self.learning_rate /= 10
                        pass

                    #3.2 train one epoch
                    step = self.train_one_epoch(sess, writer, epoch, step)

                    #3.3 save model
                    self.valid_once(sess, writer, epoch, step)
                    self.cur_epoch.load(epoch, sess)
                    self.current_itr_var.load(step, sess)
                    saver.save(sess, self.logs_dir + 'model.ckpt', step)

        writer.close()

def main(useless_argv):
 
    with tf.device('/gpu:0'):
        train_records, valid_records = scene_parsing.my_read_dataset(cfgs.seq_list_path, cfgs.anno_path)
        model = FCNNet(cfgs.mode, cfgs.max_epochs, cfgs.batch_size, cfgs.NUM_OF_CLASSESS, train_records, valid_records, cfgs.IMAGE_SIZE, cfgs.init_lr, cfgs.keep_prob)
        model.build()
        model.train()

if __name__ == '__main__':
    tf.app.run()

