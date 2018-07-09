#This is my FCN for predcting video sequence for origin size image first modified at 20180705 16:53.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import shutil
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader_video as dataset
from six.moves import xrange
#from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours, fit_ellipse_findContours_ori
from label_pred import *
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap

try:
    from .cfgs.config_video import FLAGS
except Exception:
    from cfgs.config_video import FLAGS


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
        pred_prob = tf.nn.softmax(conv_t3)

    return tf.expand_dims(annotation_pred_value,dim=3),tf.expand_dims(annotation_pred, dim=3), conv_t3, pred_prob


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation_value, pred_annotation, logits, pred_prob = inference(image, keep_probability)
    #get the softmax result
    pred_prob = tf.nn.softmax(logits) 
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    #Create a file to write logs.
    #filename='logs'+ FLAGS.mode + str(datetime.datetime.now()) + '.txt'
    filename="logs_%s%s.txt"%(FLAGS.mode,datetime.datetime.now())
    path_=os.path.join(FLAGS.logs_dir,filename)
    logs_file=open(path_,'w')
    logs_file.write("The logs file is created at %s\n" % datetime.datetime.now())
    logs_file.write("The mode is %s\n"% (FLAGS.mode))
    logs_file.write("The train data batch size is %d and the validation batch size is %d.\n"%(FLAGS.batch_size,FLAGS.v_batch_size))
    logs_file.write("The data is %s.\n" % (FLAGS.data_dir))
    logs_file.write("The data size is %d.\n" % (IMAGE_SIZE))
    logs_file.write("The model is ---%s---.\n" % FLAGS.logs_dir )
    
    print("Setting up image reader...")
    logs_file.write("Setting up image reader...\n")
    valid_video_records = scene_parsing.read_validation_video_data(FLAGS.data_dir)
    print('number of valid_records',len(valid_video_records))
    logs_file.write('number of valid_records %d\n' % len(valid_video_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    validation_dataset_reader = dataset.BatchDatset(valid_video_records, image_options)
    val_ori_dataset_reader = dataset.BatchDatset(valid_video_records)

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

        
    re_save_dir="%s%s" % (FLAGS.result_dir, datetime.datetime.now())
    logs_file.write("The result is save at file'%s'.\n" % re_save_dir)
    logs_file.write("The number of part visualization is %d.\n" % FLAGS.v_batch_size)

    #Check the result path if exists.
    if not os.path.exists(re_save_dir):
        print("The path '%s' is not found." % re_save_dir)
        print("Create now ...")
        os.makedirs(re_save_dir)
        print("Create '%s' successfully." % re_save_dir)
        logs_file.write("Create '%s' successfully.\n" % re_save_dir)
    logs_file.close()

    #Copy the logs to the result file
    result_logs_file = os.path.join(re_save_dir, filename)
    shutil.copyfile(path_, result_logs_file)
    re_save_dir_im = os.path.join(re_save_dir, 'images')
    re_save_dir_pred = os.path.join(re_save_dir, 'pred')
    re_save_dir_heat = os.path.join(re_save_dir, 'heatmap')
    re_save_dir_ellip = os.path.join(re_save_dir, 'ellip')
    re_save_dir_transheat = os.path.join(re_save_dir, 'transheat')
    if not os.path.exists(re_save_dir_im):
        os.makedirs(re_save_dir_im)
    if not os.path.exists(re_save_dir_pred):
        os.makedirs(re_save_dir_pred)
    if not os.path.exists(re_save_dir_heat):
        os.makedirs(re_save_dir_heat)
    if not os.path.exists(re_save_dir_ellip):
        os.makedirs(re_save_dir_ellip)
    if not os.path.exists(re_save_dir_transheat):
        os.makedirs(re_save_dir_transheat)

    count=0
    if_con=True
    accu_iou_t=0
    accu_pixel_t=0
       
    while if_con:
        count=count+1
        valid_images, valid_filename, if_con, start, end=validation_dataset_reader.next_batch_video_valid(FLAGS.v_batch_size)
        val_ori_images, _, _, _, _ = val_ori_dataset_reader.next_batch_video_valid(FLAGS.v_batch_size)
        pred_value, pred, logits_, pred_prob_=sess.run([pred_annotation_value, pred_annotation, logits, pred_prob],feed_dict={image: valid_images, keep_probability: 1.0})
        print("Turn %d :----start from %d ------- to %d" % (count, start, end))
        pred = np.squeeze(pred, axis=3)
        pred_value=np.squeeze(pred_value,axis=3)

 
        for itr in range(len(pred)):
            filename = valid_filename[itr]['filename']
            valid_images_ = pred_ori_visualize(val_ori_images[itr].copy(), pred[itr])
            utils.save_image(valid_images_.astype(np.uint8), re_save_dir_im, name="inp_" + filename)
            
            if FLAGS.pred:
                utils.save_gray_image(np.expand_dims(pred[itr],axis=2).astype(np.uint8), re_save_dir_pred, name="pred_" + filename)
            if FLAGS.fit_ellip:
                #valid_images_ellip=fit_ellipse_findContours_ori(valid_images[itr].copy(),np.expand_dims(pred[itr],axis=2).astype(np.uint8))
                valid_images_ellip=fit_ellipse_findContours_ori(val_ori_images[itr].copy(),np.expand_dims(pred[itr],axis=2).astype(np.uint8))
                utils.save_image(valid_images_ellip.astype(np.uint8), re_save_dir_ellip, name="ellip_" + filename)
            if FLAGS.heatmap:
                heat_map = density_heatmap(pred_prob_[itr, :, :, 1])
                utils.save_image(heat_map.astype(np.uint8), re_save_dir_heat, name="heat_" + filename)
            if FLAGS.trans_heat:
                trans_heat_map = translucent_heatmap(valid_images[itr], heat_map.astype(np.uint8).copy())
                utils.save_image(trans_heat_map, re_save_dir_transheat, name="trans_heat_" + filename)

    #logs_file.close()

if __name__ == "__main__":
    tf.app.run()
