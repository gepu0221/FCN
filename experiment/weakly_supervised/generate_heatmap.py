#This code is used to generate heatmap of the softmax probility map at 2018/05/14 20:49
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colour import Color
import pdb

#generate the heatmap
def density_heatmap(image, density_range = 1000):
    #density_range = 1000
    #numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    gradient = np.linspace(0, 1, density_range)
    im_w = image.shape[1]
    im_h = image.shape[0]
    density_map = np.zeros((im_h, im_w))
    color_map = np.empty([im_h, im_w, 3], dtype=int)
    #get gradient color using rainbow
    #use matplotlib to get color gradient
    cmap = plt.get_cmap("rainbow")
    #Use Color to generate color gradient
    blue = Color("blue")
    hex_colors = list(blue.range_to(Color("red"), density_range))
    rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]
    density_range_ = density_range - 1
    for i in range(im_h):
        for j in range(im_w):
            prob_ = int(image[i][j]*1000)
            prob_ = min(density_range_, prob_)
            for k in range(3):
                color_map[i][j][k] = rgb_colors[prob_][k]

    return color_map

#generate the heatmap(low--->high: blue--->red)
def density_heatmap_br(image, density_range = 1000):
    #density_range = 1000
    #numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    gradient = np.linspace(0, 1, density_range)
    im_w = image.shape[1]
    im_h = image.shape[0]
    density_map = np.zeros((im_h, im_w))
    color_map = np.empty([im_h, im_w, 3], dtype=int)
    #get gradient color using rainbow
    #use matplotlib to get color gradient
    cmap = plt.get_cmap("rainbow")
    #Use Color to generate color gradient
    red = Color("red")
    hex_colors = list(red.range_to(Color("blue"), density_range))
    rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]
    for i in range(im_h):
        for j in range(im_w):
            prob_ = int(image[i][j]*1000)
            #print("(%d, %d), %d" % (i, j, prob_))
            for k in range(3):
                color_map[i][j][k] = rgb_colors[prob_][k]

    return color_map



#Use the heatmap and origin image to generate translucent heatmap.
def translucent_heatmap(ori_im, heatmap, alpha = 0.3):
    overlay = ori_im.copy()
    # set the transparency
    #alpha = 0.3
    sz = ori_im.shape
    #set blue as the base heatmap color
    cv2.rectangle(overlay, (0, 0), (sz[1], sz[0]), (255,0,0), -1)
    #Overlay the base heatmap to the ori image.
    #cv2.addWeighted(overlay, alpha, ori_im, 1-alpha, 0, ori_im)
    #Overlay the heat map to the oir image.
    cv2.addWeighted(heatmap, alpha, ori_im, 1-alpha, 0, ori_im)
    
    return ori_im


if __name__ == "__main__":
    #im = cv2.imread('test.jpg')
    #sz_ = im.shape
    #prob_ = np.random.random((sz_[0], sz_[1]))
    #prob_ = np.zeros((sz_[0], sz_[1]))
    prob_ = np.linspace(0, 1, 40000)
    prob_ = prob_.reshape((200, 200))
    #im_trans = translucent_heatmap(im, prob_.astype(np.uint8))
    im_trans = density_heatmap(prob_)
    cv2.imwrite('heatmap.bmp', im_trans)
