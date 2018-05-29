import numpy as np

def polar_transform(cx, cy, fz):
    #angle, distance
    w = fz[0]
    h = fz[1]
    polar_coord = np.zeros((w, h, 2))
   
    if cx<0 or cy<0 or cx>=w or cy>=h:
        return 

    for i in range(w):
        for j in range(h):
            rx = i - cx
            ry = j - cy
            r = np.sqrt(np.power(rx,2) + np.power(ry,2))
            
            angle = np.arctan2(ry, rx)
            #print("angle : %g" % angle)
            while angle < 0:
                angle = angle + 2*np.pi

            polar_coord[i][j][0] = angle
            polar_coord[i][j][1] = r

    return polar_coord


def polar_transform_pixel(cx, cy, x, y):
    rx = x - cx
    ry = y - cy
    r = np.sqrt(np.power(rx,2) + np.power(ry,2))
    angle = np.arctan2(ry,rx)
    while angle < 0:
        angle = angle + 2*np.pi
    angle = angle / (2*np.pi) *360
    return angle, r
