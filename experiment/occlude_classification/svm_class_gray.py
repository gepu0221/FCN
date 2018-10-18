import cv2
import numpy as np
import os
import pdb
from sklearn.svm import SVC
from sklearn.externals import joblib
import glob
import time


def train():
    data_path = 'data'
    file_glob = os.path.join(data_path, 'train', 'image', '*.bmp')
    im_list = []
    im_list.extend(glob.glob(file_glob))
    data_list = []
    label_list = []

    for im in im_list:
        fn = os.path.splitext(im.split('/')[-1])[0]
        label_fn = os.path.join(data_path, 'train', 'annotation', fn+'.bmp')
        if not os.path.exists(label_fn):
            print('%s not found. Skip!' % fn)
            continue
        im = cv2.imread(im, 0)
        im_sz = im.shape
        rate = 4
        #im = cv2.resize(im, (int(im_sz[1]/rate), int(im_sz[0]/rate)), interpolation=cv2.INTER_CUBIC)
        label = cv2.imread(label_fn)
        #label = cv2.resize(label,(int(im_sz[1]/rate), int(im_sz[0]/rate)), interpolation=cv2.INTER_CUBIC) 
        cv2.imwrite('label.bmp', label)
        label = label[:, :, 0]/255

        im  = np.reshape(im, (-1, 1))
        label = np.reshape(label,(-1)).astype(np.int32)
        data_list.extend(list(im))
        label_list.extend(list(label))

    data_set = np.array(data_list)
    label_set = np.array(label_list) 
    
    t1 = time.time()
    print('-----train-------')
    svc = SVC(kernel='rbf', degree=3, gamma='auto', max_iter=-1)
    svc.fit(data_set, label_set)
    print('Train time is %g.' % (time.time() - t1))

    #Save model
    #model_n = 'train_model1017.m'
    model_n = 'train_model_gray1018.m'
    root_path = 'model'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    save_path = os.path.join(root_path, model_n)
    joblib.dump(svc, save_path)

    #return svc

def predict():

    print('-------predict---------')
    
    #Load model
    #model_n = 'train_model1017.m'
    #model_n = 'train_model1016.m'
    model_n = 'train_model_gray1018.m'
    root_path = 'model'
    load_path = os.path.join(root_path, model_n)
    if not os.path.exists(load_path):
        raise Exception('No model %s found.' % load_path)

    svc = joblib.load(load_path)

    data_path = 'data'
    file_glob = os.path.join(data_path, 'valid', 'image', '*.bmp')
    im_list = []
    im_list.extend(glob.glob(file_glob))

    for im in im_list:
        fn = os.path.splitext(im.split('/')[-1])[0]
        pred_fn = os.path.join(data_path, 'valid', 'predict', fn+'_1018.bmp')
       
        im = cv2.imread(im, 0)
        sz = im.shape
        rate = 4
        #im = cv2.resize(im, (int(sz[1]/rate), int(sz[0]/rate)), interpolation=cv2.INTER_CUBIC)

        im  = np.reshape(im, (-1, 1))
        pred = svc.predict(im)
        #pred = np.reshape(pred, (int(sz[0]/rate), int(sz[1]/rate)))
        pred = np.reshape(pred, (sz[0], sz[1]))
        cv2.imwrite(pred_fn, pred*255)

def show():
    
    data_path = 'data'
    file_glob = os.path.join(data_path, 'valid', 'predict', '*.bmp')
    im_list = []
    im_list.extend(glob.glob(file_glob))
    
    for im in im_list:
        fn = os.path.splitext(im.split('/')[-1])[0]
        pred_fn = os.path.join(data_path, 'valid', 'predict', fn+'_255''.bmp')
       
        im = cv2.imread(im)
        sz = im.shape
        rate = 1
        im = cv2.resize(im, (int(sz[1]/rate), int(sz[0]/rate)), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(pred_fn, im*255)
    

def main():
    train()
    predict()


if __name__ == '__main__':
    main()
    #show()
    


