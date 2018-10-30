import numpy as np
import cv2
import pdb
import glob
import os
from part_fintune import adpThreshold_im, otsu_threshold_
from find_nearest import find_nearest_im
from find_max_grad import find_grad_im
z=122
cc=99
s=115
x=120
q=113
w=119
e=101
r=114
p=112
o=111

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

def get_rect(ellipse_info):

    cx = ellipse_info[0][0] 
    cy = ellipse_info[0][1] 
    w = ellipse_info[1][0] 
    h = ellipse_info[1][1] 
    
    lx = int(cx - w/2)
    ly = int(cy - h/2)
    rx = int(cx + w/2)
    ry = int(cy + h/2)

    return lx, ly, rx, ry

def update_ellip(x_off, y_off, w_off, h_off, ellipse_info):
    
    cx = ellipse_info[0][0] + x_off
    cy = ellipse_info[0][1] + y_off
    w = ellipse_info[1][0] + w_off
    h = ellipse_info[1][1] + h_off

    ellipse_info = ((cx, cy), (w, h), 0)
   
    return ellipse_info


def main():
    
    root_file = 's6'
    root_dir = '%s/Image' % root_file
    txt_root_dir = root_file
    #re_root_dir = '%s_re_grad_sqrt' % root_file

    f_txt = open(os.path.join(txt_root_dir, 'groundtruth_.txt'))
    save_path =  os.path.join(txt_root_dir, 'groundtruth_correct.txt')
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            pass
    #correct_txt = open(os.path.join(txt_root_dir, 'groundtruth_correct_test_.txt'), 'w')
    
    #if not os.path.exists(re_root_dir):
        #os.makedirs(re_root_dir)

    for line in f_txt:

        f = line.split(' ')[0]
        filename = os.path.splitext(f.split("/")[-1])[0]
        f = os.path.join(root_dir, '%s.bmp' % filename)
        im = cv2.imread(f)
        ellipse_info = read_line(line)
        
        c=0
        c = cv2.waitKey(0)
        
        while c != p and c != o:

            show = im.copy()            
            x_off = y_off = w_off = h_off = 0
            
            if c == z:
                x_off = -1
            elif c == cc:
                x_off = 1
            elif c == s:
                y_off = -1
            elif c == x:
                y_off = 1
            elif c == q:
                w_off = 1
            elif c == w:
                w_off = -1
            elif c == e:
                h_off = 1
            elif c == r:
                h_off = -1
            
            ellipse_info = update_ellip(x_off, y_off, w_off, h_off, ellipse_info)
            cv2.ellipse(show, ellipse_info, (0,255,0), 1)

            cv2.imshow(filename, show)
            #pdb.set_trace()
            c=cv2.waitKey(0)
            #print(c)
            

        if c == p:
            lx, ly, rx, ry = get_rect(ellipse_info)
            new_line =  '%s %d %d %d %d %d\n' % (f, lx, ly, rx, ry, 0)
            with open(save_path, 'a') as correct_txt:
                correct_txt.write(new_line)
            cv2.destroyWindow(filename)
        if c == o:
            break
    #correct_txt.close()
    f_txt.close()
            

if __name__ == '__main__':
    main()
