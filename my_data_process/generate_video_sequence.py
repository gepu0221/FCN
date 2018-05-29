#This code is used to generate a image sequence from a video.

import tensorflow as tf
import cv2
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_video_dir', default='~/repos/data/video', help='The root folder path of the video.')
    parser.add_argument('--video_name', help='The video name')
    parser.add_argument('--root_save_dir',  default='~/repos/data/video_seq', help='The root folder path to save the image sequence.')
    parser.add_argument('--single_video', action='store_true', help='if a single video to process')
    parser.add_argument('--resize_ratio',default='0.5', help='The resize radio of saved images sequence.' )
    args = parser.parse_args()

    if not args.single_video:
        videos_list = [v for v in os.listdir(args.root_video_dir)]
        for v in videos_list:
            path_ = os.path.join(args.root_video_dir, v)
            video_name = v.split('.')[0]
            sava_path = os.path.join(args.root_save_dir, video_name)
            #Check the path if exists.
            if not os.path.exists(save_path):
                print("The save path '%s' is not found" % save_path)
                print("Create now..")
                os.makedirs(save_path)
                print("Save path create successfully!")
            cap = cv2.VideoCapture(path_)
            #frames_num = cap.get(7)
            frames_num = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
            rval, frame = cap.read()
            size = (int(frame.shape[1] * float(args.resize_ratio)), int(frame.shape[0] * float(args.resize_ratio)))
            for i in range(int(frames_num)):
                rval,frame = cap.read()
                #if read successfully.
                if rval:
                    im = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    img_name = "%s_img%05d.jpg" % (v,i)
                    img_path = os.path.join(save_path, img_name)
                    cv2.imwrite(img_path, im)
            cap.release()

    #Generate single video.
    else:
        print('the video name is %s.' % args.video_name)
        path_ = os.path.join(args.root_video_dir, args.video_name)
        print("The videl path is %s" % path_)
        video_name = args.video_name.split('.')[0]
        save_path = os.path.join(args.root_save_dir, video_name)
        #Check the path if exists.
        if not os.path.exists(save_path):
            print("The save path '%s' is not found" % save_path)
            print("Create now..")
            os.makedirs(save_path)
            print("Save path create successfully!")
        cap = cv2.VideoCapture(path_)
        frames_num = cap.get(7)
        print("The number of video is %d" % frames_num)
        rval, frame = cap.read()
        size = (int(frame.shape[1] * float(args.resize_ratio)), int(frame.shape[0] * float(args.resize_ratio)))
        for i in range(int(frames_num)):
            rval,frame = cap.read()
            #if read successfully.
            if rval:
                im = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                img_name = "%s_img%05d.jpg" % (video_name, i)
                img_path = os.path.join(save_path, img_name)
                cv2.imwrite(img_path, im)
        cap.release()



