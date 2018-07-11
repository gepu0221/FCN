import numpy as np
import tensorflow as tf
import cv2
import os
import argparse
import random

from Tools import Queue

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list_path', help='the path of label data list', default='/home/gp/repos/data/train_data')
    parser.add_argument('-r', '--result_path', help='the path of sequence list', default='/home/gp/repos/data/seq_list')
    parser.add_argument('-g', '--gt_name', help='the name of groundtruth gt file', default='groundtruth.txt') 
    parser.add_argument('-os', '--out_seq_name', default='seq_list.txt',  help='the name of output sequence gt file')
    parser.add_argument('-n', '--seq_num', help='the num of frames of a sequence', type=int, default=4)
    args = parser.parse_args()
    
    return args

#Generate a sequence images with number 'args.seq_num'
def generate_seq(q_list, seq_num):
    str_ = ''
    for i in range(seq_num):
        #use str_.split('*')[1] ... not [0]
        str_ = '%s*%s' % (str_, q_list[i])
    return str_

#Read one file data list(like s1, s2,...)
def read_one_file(file_path, seq_num):
    f = open(file_path, 'r')
    q = Queue(seq_num)
    seq_list = []
    while True:
        line = f.readline()
        if not line:
            break
        q.enqueue(line.strip('\n'))
        if q.isfull():
            if q.showFirstNum(seq_num):
                str_ = generate_seq(q.queue, seq_num)
                str_ = '%s\n' % str_
                q.dequeue()
                seq_list.append(str_)
    f.close()

    return seq_list


def main():
    args = parse_arguments()
    
    #if list path exists
    if not os.path.exists(args.list_path):
        raise Exception('No list file %s found!!!' % args.list_path)
    list_ = [d for d in os.listdir(args.list_path)]
    #if result path exists
    if not os.path.exists(args.result_path):
        print('No result file %s found!!' % args.result_path)
        print('Create ...')
        os.makedirs(args.result_path)
        print('%s create finished.' % args.result_path)
    
    total_seq_list = []
    for i in range(len(list_)):
        gt_file = os.path.join(args.list_path, list_[i], args.gt_name)

        #single file if exists
        if not os.path.isfile(gt_file):
            raise Exception('No file %s found' % gt_file)
        total_seq_list.extend(read_one_file(gt_file, args.seq_num))
    
    random.shuffle(total_seq_list)        
    
    total_num = len(total_seq_list)
    test_num = int(0.1 * total_num)
    train_num = total_num - test_num
    train_records = total_seq_list[0:train_num]
    test_records = total_seq_list[train_num:]
    print('The number of train_records is %d and test_records is %d' % (train_num, test_num))
    train_file = '%s_%s' % ('train', args.out_seq_name)
    train_out_seq_file = os.path.join(args.result_path, train_file )
    test_file = '%s_%s' % ('validation', args.out_seq_name)
    test_out_seq_file = os.path.join(args.result_path, test_file )

    train_f_seq = open(train_out_seq_file, 'w')
    for record in train_records:
        train_f_seq.write(record)
    train_f_seq.close()

    test_f_seq = open(test_out_seq_file, 'w')
    for record in test_records:
        test_f_seq.write(record)
    test_f_seq.close()


if __name__=='__main__':
    main()


