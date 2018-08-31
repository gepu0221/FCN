import numpy as np
import cv2
import os
import glob
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', help='the path of video images and also the path of result,eg.xx/xx/video_name', required=True)
    parser.add_argument('-s', '--structure', help='the structure of inside video_path', default='images/valid_video')
    parser.add_argument('-n', '--generate_name', help='the name of generate file', default='groundtruth.txt')
    parser.add_argument('-f', '--if_folder', action='store_true', help='True: video path include some videos;False:video path is a video path')
    parser.add_argument('-b', '--frame_begin', type=int, help='the begin frame of a video')
    parser.add_argument('-fn', '--frame_name', help='image name forms, if s4_img00000.jpg is s4')
    args = parser.parse_args()

    return args

#generate list of one video
def generate_one(video_path, structure_path, re_filename, begin_frame, frame_name_type):

    image_list = []
    file_glob = os.path.join(video_path, structure_path, '*.' + 'jpg')
    image_list.extend(glob.glob(file_glob)) 
    len_ = len(image_list)
    if not image_list:
        raise Exception('No file%s found' % file_glob)
    else:
        re_f = open(os.path.join(video_path, re_filename), 'w') 
        end_frame = begin_frame + len_
        for i in range(begin_frame, end_frame):
            no_ = '%05d' % i
            im_name = '%s_img%s.jpg' % (frame_name_type, no_) 
            im_path = os.path.join(video_path, structure_path, im_name)
            line = '%s %d %d %d %d %d\n' % (im_path,0,0,0,0,0)      
            re_f.write(line)
    



def main():
    args = parse_arguments()

    if args.if_folder:
        if not os.path.exists(args.video_path):
            raise Exception('No path %s found' % args.video_path)
        video_list = [v for v in os.listdir(args.video_path)]

        for v_path in video_list:
            generate_one(v_path, args.structure, args.generate_name, )
 

    else:
        generate_one(args.video_path, args.structure, args.generate_name, args.frame_begin, args.frame_name)

if __name__=='__main__':
    main()
        




