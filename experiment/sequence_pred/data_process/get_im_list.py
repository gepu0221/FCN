import glob
import os

root_path = '/home/gp/repos/FCN/experiment/sequence_pred/ellip_error'
folder = 'res_cur_seq_noda'
save_path = 'error_list'
save_file = 'train_error_list.txt'
com_path = '/home/gp/repos/data/seq_list_key_new'

def get_filename(line):
    line_s = line.split(' ')[0].split('/')
    filename = line_s[-1].split('_')[0]
    if 's4' in filename:
        im_name = line_s[-1].split('_')[1].split('part')[1]
        filename = filename+'_part'+im_name
    
    return filename

def get_filename_com(line):
    line = line.split('*')[-1]
    line_s = line.split(' ')[0].split('/')
    filename = '%s%s' % (line_s[len(line_s)-3], os.path.splitext(line_s[-1])[0])
    
    return filename


def read_compare_list(com_path):
    '''
     Read old seq_list txt to do a contrast.
     Args:
          com_path: old txt path
    '''
    file_glob = os.path.join(com_path, '*.' + 'txt')
    file_list = []
    file_list.extend(glob.glob(file_glob))
    line_map = {}

    if not file_list:
        raise Exception('No file found')

    for f_n in file_list:
        print(f_n)
        f = open(f_n, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            filename = get_filename_com(line)
            line_map[filename] = line

    return line_map

path_ = os.path.join(root_path, folder)
im_glob = os.path.join(path_, '*.bmp')
im_list = []
im_list.extend(glob.glob(im_glob))

if not os.path.exists(save_path):
    os.makedirs(save_path)

f = open(os.path.join(save_path, save_file), 'w')
line_map = read_compare_list(com_path)

if not im_list:
    raise Exception('No image list')
for im in im_list:
    filename = get_filename(im)
    print(im)
    print(filename)
    if filename == 's4':
        filename = 's4_part'
    line = line_map[filename]
    f.write(line)

f.close()
    
    
