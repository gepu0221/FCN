#This is my FCN for training first modified to add training set accurary calculation at 20180510 21:10.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
import BatchDatsetReader_soft as dataset_soft
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse,generate_heat_map,fit_ellipse_findContours

try:
    from .cfgs.config import FLAGS 
except Exception:
    from cfgs.config import FLAGS

MAX_ITERATION = FLAGS.MAX_ITERATION
NUM_OF_CLASSESS = FLAGS.NUM_OF_CLASSESS
IMAGE_SIZE = FLAGS.IMAGE_SIZE
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

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
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
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

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred_value = tf.cast(tf.subtract(tf.reduce_max(conv_t3,3),tf.reduce_min(conv_t3,3)),tf.int32)
        #annotation_pred_value = tf.argmax(conv_t3, dimension=3, name="prediction")
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred_value,dim=3),tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    soft_annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 2], name="soft_annotation")
    hard_annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="hard_annotation")

    pred_annotation_value, pred_annotation, logits = inference(image, keep_probability)
    '''
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(soft_annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    '''
    #logits:the last layer of conv net
    #labels:the ground truth
    soft_logits = tf.nn.softmax(logits/FLAGS.temperature)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits/FLAGS.temperature,
                                                                            labels = soft_annotation,
                                                                             name = "entropy_soft"))

    valid_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(hard_annotation, squeeze_dims=[3]),
                                                                          name="valid_entropy"))
   
    tf.summary.scalar("train_loss", loss)
    #tf.summary.scalar("valid_loss", valid_loss)
    tf.summary.scalar("learning_rate", learning_rate)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var, learning_rate)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    
    #Check if has the log file
    if not os.path.exists(FLAGS.logs_dir):
        print("The logs path '%s' is not found" % FLAGS.logs_dir)
        print("Create now..")
        os.makedirs(FLAGS.logs_dir)
        print("%s is created successfully!" % FLAGS.logs_dir)

    #Create a file to write logs.
    #filename='logs'+ FLAGS.mode + str(datatime.datatime.now()) + '.txt'
    filename="logs_%s%s.txt"%(FLAGS.mode,datetime.datetime.now())
    path_=os.path.join(FLAGS.logs_dir,filename)
    with open(path_,'w') as logs_file:
        logs_file.write("The logs file is created at %s.\n" % datetime.datetime.now())
        logs_file.write("The model is ---%s---.\n" % FLAGS.logs_dir)
        logs_file.write("The mode is %s\n"% (FLAGS.mode))
        logs_file.write("The train data batch size is %d and the validation batch size is %d\n."%(FLAGS.batch_size,FLAGS.v_batch_size))
        logs_file.write("The train data is %s.\n" % (FLAGS.data_dir))
        logs_file.write("The data size is %d and the MAX_ITERATION is %d.\n" % (IMAGE_SIZE, MAX_ITERATION))
        logs_file.write("Setting up image reader...")

    
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.my_read_dataset(FLAGS.data_dir)
    print('number of train_records',len(train_records))
    print('number of valid_records',len(valid_records))
    with open(path_, 'a') as logs_file:
        logs_file.write('number of train_records %d\n' % len(train_records))
        logs_file.write('number of valid_records %d\n' % len(valid_records))

    path_lr = FLAGS.learning_rate_path
    with open(path_lr, 'r') as f:
        lr_ = float(f.readline().split('\n')[0])


    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset_soft.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    #if not train,restore the model trained before
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    num_=0
    current_itr = FLAGS.train_itr
    #pre fetch valid data for tensorboard
    valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.v_batch_size)
    print(valid_annotations.shape)

    for itr in xrange(current_itr, MAX_ITERATION):

        with open(path_lr, 'r') as f:
            lr_ = float(f.readline().split('\n')[0])

        train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, soft_annotation: train_annotations/int(FLAGS.normal), keep_probability: 0.85, learning_rate: lr_}
        

        sess.run(train_op, feed_dict=feed_dict)
        logs_file = open(path_, 'a')  
        if itr % 10 == 0:
            print('learning_rate : %g' % lr_)
            train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            summary_writer.add_summary(summary_str, itr)

        if itr % 10 == 0:
            
            #Caculate the accurary at the training set.
            train_random_images, train_random_annotations = train_dataset_reader.get_random_batch_for_train(FLAGS.v_batch_size)
            train_loss,train_pred_anno = sess.run([loss,pred_annotation], feed_dict={image:train_random_images,
                                                                                        soft_annotation:train_random_annotations/int(FLAGS.normal),
                                                                                        keep_probability:1.0})
            accu_iou_,accu_pixel_ = accu.caculate_accurary(train_pred_anno, train_random_annotations)
            print("%s ---> Training_loss: %g" % (datetime.datetime.now(), train_loss))
            print("%s ---> Training_pixel_accuary: %g" % (datetime.datetime.now(),accu_pixel_))
            print("%s ---> Training_iou_accuary: %g" % (datetime.datetime.now(),accu_iou_))
            print("---------------------------")
            #Output the logs.
            num_ = num_ + 1
            logs_file.write("No.%d the itr number is %d.\n" % (num_, itr))
            logs_file.write("%s ---> Training_loss: %g.\n" % (datetime.datetime.now(), train_loss))
            logs_file.write("%s ---> Training_pixel_accuary: %g.\n" % (datetime.datetime.now(),accu_pixel_))
            logs_file.write("%s ---> Training_iou_accuary: %g.\n" % (datetime.datetime.now(),accu_iou_))
            logs_file.write("---------------------------\n")
            
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.v_batch_size)
            
            print(valid_annotations.shape)
            
            #valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       #keep_probability: 1.0})
            valid_loss_,pred_anno=sess.run([valid_loss,pred_annotation],feed_dict={image:valid_images,
                                                                                      hard_annotation:valid_annotations,
                                                                                      keep_probability:1.0})
            accu_iou,accu_pixel=accu.caculate_accurary(pred_anno,valid_annotations)
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss_))
            print("%s ---> Validation_pixel_accuary: %g" % (datetime.datetime.now(),accu_pixel))
            print("%s ---> Validation_iou_accuary: %g" % (datetime.datetime.now(),accu_iou))

            #Output the logs.
            logs_file.write("%s ---> Validation_loss: %g.\n" % (datetime.datetime.now(), valid_loss_))
            logs_file.write("%s ---> Validation_pixel_accuary: %g.\n" % (datetime.datetime.now(),accu_pixel))
            logs_file.write("%s ---> Validation_iou_accuary: %g.\n" % (datetime.datetime.now(),accu_iou))
           
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            #End the iterator
        logs_file.close()
if __name__ == "__main__":
    tf.app.run()
