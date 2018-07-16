#This is my FCN for sequence predction on video without annotations first modified at 20180712.

from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import TensorflowUtils as utils
import read_data as scene_parsing
import datetime
import BatchReader as dataset
import CaculateAccurary as accu
from six.moves import xrange
from label_pred import pred_visualize, anno_visualize, fit_ellipse, generate_heat_map, fit_ellipse_findContours
from generate_heatmap import density_heatmap, density_heatmap_br, translucent_heatmap
import shutil

try:
    from .cfgs.config_pred import cfgs
except Exception:
    from cfgs.config_pred import cfgs

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = cfgs.NUM_OF_CLASSESS
IMAGE_SIZE = cfgs.IMAGE_SIZE


def vgg_net(weights, image):
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
            if cfgs.debug:
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
    model_data = utils.get_model_data(cfgs.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    #processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        W1 = utils.weight_variable([3, 3, 7, 64], name="W1")
        b1 = utils.bias_variable([64], name="b1")
        conv1 = utils.conv2d_basic(image, W1, b1)
        relu1 = tf.nn.relu(conv1, name='relu1')
        
        #pretrain
        image_net = vgg_net(weights, relu1)
        
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if cfgs.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if cfgs.debug:
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



def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 7], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation_value, pred_annotation, logits, pred_prob = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    #logits:the last layer of conv net
    #labels:the ground truth
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if cfgs.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    #Create a file to write logs.
    #filename='logs'+ cfgs.mode + str(datetime.datetime.now()) + '.txt'
    filename="logs_%s%s.txt"%(cfgs.mode,datetime.datetime.now())
    path_=os.path.join(cfgs.logs_dir,filename)
    logs_file=open(path_,'w')
    logs_file.write("The logs file is created at %s\n" % datetime.datetime.now())
    logs_file.write("The mode is %s\n"% (cfgs.mode))
    logs_file.write("The train data batch size is %d and the validation batch size is %d.\n"%(cfgs.batch_size,cfgs.v_batch_size))
    logs_file.write("The train data is %s.\n" % (cfgs.data_dir))
    logs_file.write("The model is ---%s---.\n" % cfgs.logs_dir )
    
    print("Setting up image reader...")
    logs_file.write("Setting up image reader...\n")
    train_records, valid_records = scene_parsing.my_read_video_dataset(cfgs.seq_list_path, cfgs.anno_path)
    print('number of train_records',len(train_records))
    print('number of valid_records',len(valid_records))
    logs_file.write('number of train_records %d\n' % len(train_records))
    logs_file.write('number of valid_records %d\n' % len(valid_records))

    print("Setting up dataset reader")
    vis = True if cfgs.mode == 'all_visualize' else False
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE, 'visualize': vis}
        
    if cfgs.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(cfgs.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(cfgs.logs_dir)
    #if not train,restore the model trained before
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

        
    if cfgs.mode == "accurary":
        count=0
        if_con=True
        accu_iou_t=0
        accu_pixel_t=0
        
        while if_con:
            count=count+1
            valid_images, valid_annotations, valid_filenames, if_con, start, end=validation_dataset_reader.next_batch_valid(cfgs.v_batch_size)
            valid_loss,pred_anno=sess.run([loss,pred_annotation],feed_dict={image:valid_images,
                                                                                      annotation:valid_annotations,
                                                                                      keep_probability:1.0})
            accu_iou,accu_pixel=accu.caculate_accurary(pred_anno,valid_annotations)
            print("Ture %d ---> the data from %d to %d" % (count, start, end))
            print("%s ---> Validation_pixel_accuary: %g" % (datetime.datetime.now(),accu_pixel))
            print("%s ---> Validation_iou_accuary: %g" % (datetime.datetime.now(),accu_iou))
            #Output logs.
            logs_file.write("Ture %d ---> the data from %d to %d\n" % (count, start, end))
            logs_file.write("%s ---> Validation_pixel_accuary: %g\n" % (datetime.datetime.now(),accu_pixel))
            logs_file.write("%s ---> Validation_iou_accuary: %g\n" % (datetime.datetime.now(),accu_iou))

            accu_iou_t=accu_iou_t+accu_iou
            accu_pixel_t=accu_pixel_t+accu_pixel
        print("%s ---> Total validation_pixel_accuary: %g" % (datetime.datetime.now(),accu_pixel_t/count))
        print("%s ---> Total validation_iou_accuary: %g" % (datetime.datetime.now(),accu_iou_t/count))
        #Output logs
        logs_file.write("%s ---> Total validation_pixel_accurary: %g\n" % (datetime.datetime.now(),accu_pixel_t/count))
        logs_file.write("%s ---> Total validation_iou_accurary: %g\n" % (datetime.datetime.now(),accu_iou_t/count))

    elif cfgs.mode == "all_visualize":
           
        re_save_dir="%s%s" % (cfgs.result_dir, datetime.datetime.now())
        logs_file.write("The result is save at file'%s'.\n" % re_save_dir)
        logs_file.write("The number of part visualization is %d.\n" % cfgs.v_batch_size)
       
        #Check the result path if exists.
        if not os.path.exists(re_save_dir):
            print("The path '%s' is not found." % re_save_dir)
            print("Create now ...")
            os.makedirs(re_save_dir)
            print("Create '%s' successfully." % re_save_dir)
            logs_file.write("Create '%s' successfully.\n" % re_save_dir)
       
        re_save_dir_im = os.path.join(re_save_dir, 'images')
        re_save_dir_heat = os.path.join(re_save_dir, 'heatmap')
        re_save_dir_ellip = os.path.join(re_save_dir, 'ellip')
        re_save_dir_transheat = os.path.join(re_save_dir, 'transheat')
        if not os.path.exists(re_save_dir_im):
            os.makedirs(re_save_dir_im)
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
            valid_images, valid_filename, valid_cur_images, if_con, start, end=validation_dataset_reader.next_batch_video_valid(cfgs.v_batch_size)
            pred_value, pred, logits_, pred_prob_=sess.run([pred_annotation_value, pred_annotation, logits, pred_prob],feed_dict={image: valid_images, keep_probability: 1.0})
            print("Turn %d :----start from %d ------- to %d" % (count, start, end))
            pred = np.squeeze(pred, axis=3)
            pred_value=np.squeeze(pred_value,axis=3)

 
            for itr in range(len(pred)):
                filename = valid_filename[itr]['filename']
                valid_images_ = pred_visualize(valid_cur_images[itr].copy(), pred[itr])
                utils.save_image(valid_images_.astype(np.uint8), re_save_dir_im, name="inp_" + filename)
            
                if cfgs.fit_ellip:
                    #valid_images_ellip=fit_ellipse_findContours_ori(valid_images[itr].copy(),np.expand_dims(pred[itr],axis=2).astype(np.uint8))
                    valid_images_ellip=fit_ellipse_findContours(valid_cur_images[itr].copy(),np.expand_dims(pred[itr],axis=2).astype(np.uint8))
                    utils.save_image(valid_images_ellip.astype(np.uint8), re_save_dir_ellip, name="ellip_" + filename)
                if cfgs.heatmap:
                    heat_map = density_heatmap(pred_prob_[itr, :, :, 1])
                    utils.save_image(heat_map.astype(np.uint8), re_save_dir_heat, name="heat_" + filename)
                if cfgs.trans_heat:
                    trans_heat_map = translucent_heatmap(valid_cur_images[itr], heat_map.astype(np.uint8).copy())
                    utils.save_image(trans_heat_map, re_save_dir_transheat, name="trans_heat_" + filename)
   
if __name__ == "__main__":
    tf.app.run()
