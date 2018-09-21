import numpy as np
import cv2
import pdb
import glob
import os
from part_fintune import adpThreshold_im

def read_line(line):
    line_s = line.split(' ')

    lx, ly, rx, ry = int(line_s[1]), int(line_s[2]), int(line_s[3]), int(line_s[4])
    w = rx - lx
    h = ry - ly
    cx=(lx+rx)/2
    cy=(ly+ry)/2

    ellipse_info = ((cx, cy), (w, h), 0)

    return ellipse_info

def main_adap():
    
    root_dir = 's6'
    re_root_dir = 's6_re'
    glob_file = os.path.join(root_dir, '*.bmp')
    file_list = []
    file_list.extend(glob.glob(glob_file))
    
    f_txt = open(os.path.join(root_dir, 'groundtruth.txt'))
    
    if not os.path.exists(re_root_dir):
        os.makedirs(re_root_dir)

    for f in file_list:
        filename = os.path.splitext(f.split("/")[-1])[0]
        im = cv2.imread(filename)
        im_gray = cv2.imread(filename, 0)
        line = f_txt.readline()
        ellipse_info = read_line(line)

        after_adap = adpThreshold_im(im, im_gray, ellipse_info)
        
        re_filename = os.path.join(re_root_dir, 'adaped_%s.bmp' % filename)
        cv2.imwrite(re_filename, after_adap)


        

if __name__ == '__main__':
    
    main_adap()
