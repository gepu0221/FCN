import numpy as np
import cv2
import pdb

filename = 'img00103'

def translucent_im(src_im, overlay_im, alpha = 0.3):
    
    overlay = src_im.copy()
    sz = src_im.shape
    cv2.rectangle(overlay, (0, 0), (sz[1], sz[0]), (255, 0, 0), -1)
    cv2.addWeighted(overlay_im, alpha, src_im, 1-alpha, 0, src_im)

    return src_im

def get_marks(im):

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=2)
    #gray = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=2)
    #gray = cv2.Canny(gray, threshold1=20 , threshold2=150)
    gray = cv2.Canny(gray, threshold1=20 , threshold2=100)

    _,contours,hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sz = gray.shape
    marks = np.zeros((sz[0], sz[1]))
    im_contours = np.zeros((sz[0], sz[1]))
    c_len = len(contours)
    for i in range(c_len):
        color = (i, i, i)
        cv2.drawContours(marks, contours, i, color, thickness=3)
    cv2.drawContours(im_contours, contours, -1, (255,255,255), thickness=3)

    return marks


def watershed(im3):
    
    marks = get_marks(im3)
    cv2.imwrite('%s_m.bmp' % filename, marks)
    marks = marks.astype(np.int32)
    marks_after = cv2.watershed(im3, marks)

    sz = marks_after.shape
    marks_o = np.zeros((sz[0], sz[1], 3))
    marks_o[:,:,0] = marks_after
    marks_o[:,:,1] = marks_after
    marks_o[:,:,2] = marks_after
    im_transl = translucent_im(im3, marks_o.astype(np.uint8), 0.5)
    cv2.imwrite('%s_transl.bmp' % filename, im_transl)

    return marks_after
    #return marks_o


def main():
    
    im = cv2.imread('%s.bmp' % filename, 1)
    im_after = watershed(im)

    cv2.imwrite('%s_re.bmp' % filename, im_after)

if __name__ == '__main__':
    main()
    

   
