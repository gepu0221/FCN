import cv2
import os
import math
import pdb
import numpy as np

UNKNOWN_FLOW_THRESH = 1e9
NOTABAL_FLOW_THRESH = 1

def makecolorwheel():
    
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    colorwheel = []

    for i in range(RY):
        colorwheel.append(np.array((255, 255*i/RY, 0)))
    for i in range(YG):
        colorwheel.append(np.array((255-255*i/YG, 255, 0)))
    for i in range(GC):
        colorwheel.append(np.array((0, 255, 255*i/GC)))
    for i in range(CB):
        colorwheel.append(np.array((0, 255-255*i/CB, 255)))
    for i in range(BM):
        colorwheel.append(np.array((255*i/BM, 0, 255)))
    for i in range(MR):
        colorwheel.append(np.array((255, 0, 255-255*i/MR)))

    return colorwheel


def flowToColor(flow):
    '''
    Args: show flow color map.
    '''
    sz = flow.shape
    color = np.zeros((sz[0], sz[1], 3))

    colorwheel = makecolorwheel()

    maxrad = -1

    for i in range(sz[0]):
        for j in range(sz[1]):
            fx = flow[i][j][0]
            fy = flow[i][j][1]
            #if abs(fx)>NOTABAL_FLOW_THRESH or abs(fy)>NOTABAL_FLOW_THRESH:
            #    print('(%d, %d) ' % (i, j), 'fx, fy ', flow[i][j])  
            if abs(fx)>UNKNOWN_FLOW_THRESH or abs(fy)>UNKNOWN_FLOW_THRESH:
                continue

            rad = math.sqrt(fx*fx + fy*fy)
            if maxrad < rad:
                maxrad = rad

    for i in range(sz[0]):
        for j in range(sz[1]):
            fx = flow[i][j][0]
            fy = flow[i][j][1]

            if abs(fx)>UNKNOWN_FLOW_THRESH or abs(fy)>UNKNOWN_FLOW_THRESH:
                color[i][j] = 0
                continue

            rad = math.sqrt(fx*fx + fy*fy)
            angle = math.atan2(-fy, -fx) / math.pi
            fk = (angle+1.0) / 2.0 * (len(colorwheel)-1)
            k0 = int(fk)
            k1 = (k0+1) % len(colorwheel)
            f = float(fk - k0)

            for b in range(3):
                col0 = colorwheel[k0][b] / 255.0
                col1 = colorwheel[k0][b] / 255.0
                col = (1-f) * col0 + f * col1
                if rad <= 1:
                    col = 1- rad * (1-col)
                else:
                    col *= 0.75

                color[i][j][2-b] = int(255.0*col)

    return color


def med_flow_color(flow):
    '''
        medianBlur the flow before color.
    '''
    flow_x = (flow[:, :, 0] * 1000).astype(np.uint8)
    flow_y = (flow[:, :, 1] * 1000).astype(np.uint8)

    flow[:, :, 0] = cv2.medianBlur(flow_x, 3).astype(np.float32) / 1000
    flow[:, :, 1] = cv2.medianBlur(flow_y, 3).astype(np.float32) / 1000

    color = flowToColor(flow)

    return color

def extract_pair_flow(pre_im, cur_im):
    '''
    Args: Ectract optical flow of a pair of images.
    '''
    if pre_im is None or cur_im is None:
        raise Exception('Gray data is error!!')

    #check if gray image.
    sz_pre = pre_im.shape
    if sz_pre[2] == 3:
        pre_im = cv2.cvtColor(pre_im, cv2.COLOR_BGR2GRAY)
    sz_cur = cur_im.shape
    if sz_cur[2] == 3:
        cur_im = cv2.cvtColor(cur_im, cv2.COLOR_BGR2GRAY)

    
    flow = cv2.calcOpticalFlowFarneback(pre_im, cur_im, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #print(flow)   
    color = flowToColor(flow)

    return color

def main():
    pre_name = 'img00174.bmp'
    cur_name = 'img00190.bmp'
    
    pre_im = cv2.imread(pre_name)
    cur_im = cv2.imread(cur_name)

    color = extract_pair_flow(pre_im, cur_im)
    cv2.imwrite('flow190.bmp', color)

if __name__ == "__main__":
    main()
