import os
import numpy as np

src_f = 'data_list/s8_val_gen_check1201.txt'
dst_f = 'data_list/s8_val_gen_sort1202_.txt'

dict_ = {}

f = open(src_f)
f_d = open(dst_f, 'w')

for line in f:
    path = line.split(' ')[0]
    fn = os.path.splitext(path.split("/")[-1])[0]
    num = int(fn.split('img')[1])
    dict_[num] = line
f.close()

for i in range(10000,19999):
    line = dict_[i]
    f_d.write(line)
f_d.close()
