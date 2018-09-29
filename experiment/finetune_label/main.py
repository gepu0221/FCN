import numpy as np
import cv2
import pdb
import glob
import os
import time
from part_fintune import adpThreshold_im, otsu_threshold_
from find_nearest import find_nearest_im
from find_max_grad import find_grad_im, find_grad_adap_im, find_grad_im_time, get_dis_map, get_dis_map4

def read_line(line):
    line_s = line.split(' ')
    print(line_s)
    lx, ly, rx, ry = int(float(line_s[1])), int(float(line_s[2])), int(float(line_s[3])), int(float(line_s[4].split('\n')[0]))
    w = rx - lx
    h = ry - ly
    cx=(lx+rx)/2
    cy=(ly+ry)/2

    ellipse_info = ((cx, cy), (w, h), 0)
   
    return ellipse_info

def main_adap():
    
    root_file = 's6'
    root_dir = '%s/Image' % root_file
    txt_root_dir = root_file
    re_root_dir = '%s_re' % root_file
    glob_file = os.path.join(root_dir, '*.bmp')
    file_list = []
    file_list.extend(glob.glob(glob_file))

    f_txt = open(os.path.join(txt_root_dir, 'groundtruth.txt'))
    
    if not os.path.exists(re_root_dir):
        os.makedirs(re_root_dir)

    #for f in file_list:
    for line in f_txt:
        f = line.split(' ')[0]
        filename = os.path.splitext(f.split("/")[-1])[0]
        f = os.path.join(root_dir, '%s.bmp' % filename)
        im = cv2.imread(f)
        im_gray = cv2.imread(f, 0)
        ellipse_info = read_line(line)
        after_adap = adpThreshold_im(im, im_gray, ellipse_info)
        
        #draw ellipse 
        cv2.ellipse(after_adap, ellipse_info, (0,255,0), 1)

        re_filename = os.path.join(re_root_dir, 'adaped_%s_0.bmp' % filename)
        cv2.imwrite(re_filename, after_adap)

def main_nearest():
    
    root_file = 's8'
    root_dir = '%s/Image' % root_file
    txt_root_dir = root_file
    re_root_dir = '%s_re_near_new_comp' % root_file
    glob_file = os.path.join(root_dir, '*.bmp')
    file_list = []
    file_list.extend(glob.glob(glob_file))

    f_txt = open(os.path.join(txt_root_dir, 'groundtruth.txt'))
    
    if not os.path.exists(re_root_dir):
        os.makedirs(re_root_dir)

    #for f in file_list:
    for line in f_txt:
        f = line.split(' ')[0]
        filename = os.path.splitext(f.split("/")[-1])[0]
        f = os.path.join(root_dir, '%s.bmp' % filename)
        im = cv2.imread(f)
        im_gray = cv2.imread(f, 0)
        ellipse_info = read_line(line)
        after_nearest, im_show = find_nearest_im(im, im_gray, ellipse_info)
        
        #draw ellipse 
        #cv2.ellipse(after_nearest, ellipse_info, (0,255,0), 1)

        re_filename = os.path.join(re_root_dir, 'near_%s_20.bmp' % filename)
        cv2.imwrite(re_filename, after_nearest)
        show_filename = os.path.join(re_root_dir, 'near_%s_show20_no.bmp' % filename)
        cv2.imwrite(show_filename, im_show)


def main_max_grad():
    
    root_file = 's8'
    root_dir = '%s/Image' % root_file
    txt_root_dir = root_file
    re_root_dir = '%s_re_grad_test_time0929' % root_file
    glob_file = os.path.join(root_dir, '*.bmp')
    file_list = []
    file_list.extend(glob.glob(glob_file))

    f_txt = open(os.path.join(txt_root_dir, 'groundtruth.txt'))
    
    if not os.path.exists(re_root_dir):
        os.makedirs(re_root_dir)

    count = -1
    sz = [40, 40]
    dis_map = get_dis_map(sz)

    for line in f_txt:

        count += 1
        if count < 38:
            continue
        f = line.split(' ')[0]
        filename = os.path.splitext(f.split("/")[-1])[0]
        f = os.path.join(root_dir, '%s.bmp' % filename)
        im = cv2.imread(f)
        im_gray = cv2.imread(f, 0)
        ellipse_info = read_line(line)
        after_grad, im_show = find_grad_im(im, im_gray, ellipse_info)
        #after_grad, im_show = find_grad_adap_im(im, im_gray, ellipse_info)

        re_filename = os.path.join(re_root_dir, 'grad_%s_adap20_sqrt.bmp' % filename)
        cv2.imwrite(re_filename, after_grad)
        show_filename = os.path.join(re_root_dir, 'grad_%s_show20.bmp' % filename)
        #cv2.imwrite(show_filename, im_show)

        #draw ellipse 
        cv2.ellipse(after_grad, ellipse_info, (0,255,0), 1)
        
        comp_filename = os.path.join(re_root_dir, 'grad_%s_adap20_sqrt_comp.bmp' % filename)
        cv2.imwrite(comp_filename, after_grad)

def main_test_time():
   
   root_file = 's8'
   root_dir = '%s/Image' % root_file
   txt_root_dir = root_file
   re_root_dir = '%s_re_grad_test_time' % root_file
   glob_file = os.path.join(root_dir, '*.bmp')
   file_list = []
   file_list.extend(glob.glob(glob_file))

   f_txt = open(os.path.join(txt_root_dir, 'groundtruth.txt'))
   
   if not os.path.exists(re_root_dir):
       os.makedirs(re_root_dir)

   count = -1
   sz = [40, 40]
   dis_map = get_dis_map4(sz)

   for line in f_txt:

       count += 1
       if count < 38:
           continue
       f = line.split(' ')[0]
       filename = os.path.splitext(f.split("/")[-1])[0]
       f = os.path.join(root_dir, '%s.bmp' % filename)
       im = cv2.imread(f)
       im_gray = cv2.imread(f, 0)
       ellipse_info = read_line(line)
       t1 = time.time()
       im_cc = find_grad_im_time(im, im_gray, ellipse_info)
       print('time %g of a image' % (time.time()-t1))

       re_filename = os.path.join(re_root_dir, 'im_grad_%s.bmp' % filename)
       cv2.imwrite(re_filename, im_cc)

       #draw ellipse 
       #cv2.ellipse(after_grad, ellipse_info, (0,255,0), 1)
       
       #comp_filename = os.path.join(re_root_dir, 'grad_%s_adap20_sqrt_comp.bmp' % filename)
       #cv2.imwrite(comp_filename, after_grad)

       

       

if __name__ == '__main__':
    
    #main_adap()
    #main_nearest()
    #main_max_grad()
    main_test_time()
