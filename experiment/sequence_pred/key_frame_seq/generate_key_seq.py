import tensorflow as tf
import numpy as np
import os
from six.moves import cPickle as pickle
import cv2
import argparse
import random
import glob

from Tools import Queue

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list_path', help='the path of label data list', default='/home/gp/repos/data/train_data')
    parser.add_argument('-c', '--com_path', help='the path of compare data list', default='/home/gp/repos/data/seq_list_resize') 
    parser.add_argument('-r', '--result_path', help='the path of sequence list', default='/home/gp/repos/data/seq_list_key')
    parser.add_argument('-g', '--gt_name', help='the name of groundtruth gt file', default='groundtruth.txt') 
    parser.add_argument('-m', '--map_path', help='the path of ket map pickle', default='/home/gp/repos/FCN/experiment/sequence_pred/key_frame_pickle/key_frame_pickle0814_20.pickle')
    parser.add_argument('-os', '--out_seq_name', default='seq_list.txt',  help='the name of output sequence gt file')
    parser.add_argument('-n', '--seq_num', help='the num of frames of a sequence', type=int, default=5)
    args = parser.parse_args()
    
    return args

def get_filename(line):
    line_s = line.split(' ')[0].split('/')
    filename = '%s%s' % (line_s[len(line_s)-3], os.path.splitext(line_s[-1])[0])

    return filename

def generate_seq(line_list, key_list, seq_num, index):
    '''
    Args:
        index: index of current frame in line_list
    '''
    str_ = line_list[index]
    seq_index = index-1
    count = 1
    while count<seq_num:
        if seq_index<=0:
            str_ = None
            break
        if not key_list[seq_index]:
            seq_index -= 1
            continue
        str_ = '%s*%s' % (line_list[seq_index], str_)
        count += 1
        seq_index -= 1

    return str_

def generate_seq_with_name(line_list, key_list, seq_num, index):
    '''
    Args:
        index: index of current frame in line_list
    '''
    str_ = line_list[index]
    str_c = str_
    filename = get_filename(str_)
    seq_index = index-1
    count = 1
    while count<seq_num:
        if seq_index<=0:
            str_ = ''
            for i in range(seq_num):
                str_ = '%s*%s' % (str_c, str_)
            break
        if not key_list[seq_index]:
            seq_index -= 1
            continue
        str_ = '%s*%s' % (line_list[seq_index], str_)
        count += 1
        seq_index -= 1
  
    return str_, filename


def read_one_file(file_path, seq_num, key_map):
    '''
    Read one file to generate sequence.
    Args:
        k_map: decide current frame if a key frame.Only key frame can add to sequence.
    '''
    f = open(file_path, 'r')
    seq_list = []
    line_list = []
    key_list = []
    while True:
        line = f.readline()
        if not line:
            break
        im_path_s = line.split(' ')[0].split('/')
        filename = '%s%s' % (im_path_s[len(im_path_s)-3], os.path.splitext(im_path_s[-1])[0])
        if filename in key_map:
            key_list.append(key_map[filename])
        else:
            key_list.append(True)

        line_list.append(line.strip('\n').replace('train_data', 'train_data_seq'))
    f.close()

    for i in range(len(line_list)):
        str_ = generate_seq(line_list, key_list, seq_num, i)
        if str_:
            str_ = '*%s\n' % str_
            seq_list.append(str_)

    return seq_list

def read_one_file_map(file_path, seq_num, key_map, line_map):
    '''
    Read one file to generate sequence.
    Args:
        k_map: decide current frame if a key frame.Only key frame can add to sequence.
    '''
    f = open(file_path, 'r')
    seq_list = []
    line_list = []
    key_list = []
    while True:
        line = f.readline()
        if not line:
            break
        im_path_s = line.split(' ')[0].split('/')
        filename = '%s%s' % (im_path_s[len(im_path_s)-3], os.path.splitext(im_path_s[-1])[0])
        #print(filename)
        if filename in key_map:
            key_list.append(key_map[filename])
        else:
            key_list.append(True)

        line_list.append(line.strip('\n').replace('train_data', 'train_data_seq'))
    f.close()

    for i in range(len(line_list)):
        str_, filename = generate_seq_with_name(line_list, key_list, seq_num, i)
        if str_:
            str_ = '*%s\n' % str_
            line_map[filename] = str_

    return line_map


def read_compare_list(com_path):
    '''
     Read old seq_list txt to do a contrast.
     Args:
          com_path: old txt path
    '''
    file_glob = os.path.join(com_path, '*.' + 'txt')
    file_list = []
    file_list.extend(glob.glob(file_glob))
    line_list = []

    if not file_list:
        raise Exception('No file found')

    for f_n in file_list:
        print(f_n)
        f = open(f_n, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            line_list.append(line)
            #print(line)
        f.close()

    return line_list

def compare(old_line_list, new_line_map, seq_num):
    len_ = len(old_line_list)
    total_seq_list = []

    for i in range(len_):
        line = old_line_list[i]
        line = line.split('*')
        line_cur_s = line[seq_num]
        cur_name = get_filename(line_cur_s)
        if cur_name in new_line_map:
            total_seq_list.append(new_line_map[cur_name])
        else:
            print('Not found %s' % cur_name)

    return total_seq_list
            

def main():
    args = parse_arguments()
    
    #if pickel map data path exists
    if not os.path.exists(args.map_path):
        raise Exception('No key map pickek file %s found!!!' % args.map_path)
  
    with open(args.map_path, 'rb') as f:
        key_map = pickle.load(f)
        
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
    
  
    for i in range(len(list_)):
        gt_file = os.path.join(args.list_path, list_[i], args.gt_name)

        #single file if exists
        if not os.path.isfile(gt_file):
            raise Exception('No file %s found' % gt_file)
        total_seq_list.extend(read_one_file(gt_file, args.seq_num, key_map))
    
    random.shuffle(total_seq_list)        
    
    total_num = len(total_seq_list)
    print(total_num[0])
    #train_num = int(0 * total_num)
    #test_num = total_num - train_num
    train_num = 11808
    test_num = 1280
    test_records = total_seq_list[0:test_num]
    train_records = total_seq_list[test_num:13088]
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


def compare_main():
    args = parse_arguments()
    
    #if pickel map data path exists
    if not os.path.exists(args.map_path):
        raise Exception('No key map pickek file %s found!!!' % args.map_path)
  
    with open(args.map_path, 'rb') as f:
        key_map = pickle.load(f)
        
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
    
    com_line_list = read_compare_list(args.com_path)

    new_line_map = {}

    for i in range(len(list_)):
        gt_file = os.path.join(args.list_path, list_[i], args.gt_name)

        #single file if exists
        if not os.path.isfile(gt_file):
            raise Exception('No file %s found' % gt_file)
        read_one_file_map(gt_file, args.seq_num, key_map, new_line_map)
    
    total_seq_list = compare(com_line_list, new_line_map, args.seq_num)
    print(total_seq_list[0])

    total_num = len(total_seq_list)
    print(total_num)
    #train_num = int(0 * total_num)
    #test_num = total_num - train_num
    train_num = 11808
    test_num = 1280
    test_records = total_seq_list[0:test_num]
    train_records = total_seq_list[test_num:13088]
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
    compare_main()


            
