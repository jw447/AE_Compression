'''
the file parse the xml label file
and save both image and labels into binary files

'''

import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
from scipy import misc
from glob import glob
import multiprocessing
from joblib import Parallel, delayed
import scipy.io as sio


DATA_PATH = '../data/SyntheticScatteringData3'
DATA_PATH_BIN = DATA_PATH + 'BIN_log_32'
ALL_TAG_FILE = '../data/17tags_meta.txt'
TAG_ID = 1
ALL_TAG_META = []
with open(ALL_TAG_FILE) as f:
    ALL_TAG_META = f.read().lower().split('\n')

# ALL_TAG_META = ALL_TAG_META[TAG_ID]

NUM_TAGS = len(ALL_TAG_META)
# all tag_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
ALL_TAG_META = zip(np.arange(0, NUM_TAGS), ALL_TAG_META)
IMAGESIZE = 256
TRAIN_RATIO = 0.8

# data_file_name could be like ../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat
# ../data/SyntheticScatteringData/014267be_varied_sm/analysis/results/00000001.xml
# return a label vector of size [num_of_tags]
def parse_label(data_file_name):
    data_path = data_file_name.rsplit('/',1)
    label_file_name = data_path[0] + '/analysis/results/' + data_path[1].split('.')[0] + '.xml'
    label_vector = np.zeros([NUM_TAGS])

    if os.path.exists(label_file_name):
        root = ET.parse(label_file_name).getroot()
        for result in root[0]:
            attribute = result.attrib.get('name')
            attribute = attribute.rsplit('.', 1)
            # only care about high level tags
            attribute = attribute[1].split(':')[0]
            # check if that attribute is with the all_tag_file
            for i in range(NUM_TAGS):
                if ALL_TAG_META[i][1] == attribute:
                    label_vector[ALL_TAG_META[i][0]] = 1


    else:
        print ('%s does not exist!' %label_file_name)

    flag = bool(sum(label_vector))
    return label_vector, flag

def _binaryize_one_dir(dir):
    file_names = os.listdir(dir)
    string_binary = ''
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag = parse_label(os.path.join(dir, data_file))
            # if not flag:
            #     print(os.path.join(dir, data_file))
            # else:
            label = label.astype('int16')
            label = list(label)
            label_byte = struct.pack('h'*len(label), *label)
            string_binary += label_byte
            image = sio.loadmat(os.path.join(dir, data_file))
            # the shape of image is 256*256
            image = image['detector_image']
            image = np.reshape(image, [-1])
            # take the log
            image = np.log(image) / np.log(1.0414)
            image[np.isinf(image)] = 0
            image = image.astype('int16')
            image = list(image)
            image_byte = struct.pack('h'*len(image), *image)
            string_binary += image_byte

    return string_binary


def _binaryize_one_list(all_dirs, idx_permutation, base_num, step_size,  save_name, i):
    tmp_dir = [all_dirs[idx_permutation[j]] for j in range((i+base_num)*step_size, (i+base_num+1)*step_size)]
    print('processing %s_%d.bin' % (save_name, i))
    with open(os.path.join(DATA_PATH_BIN, '%s_%d.bin' % (save_name, i)), 'wb') as f:
        for dir_list_i in tmp_dir:
            f.write(_binaryize_one_dir(dir_list_i))


def processFile_parallel():
    dirs = glob(DATA_PATH + '/*')

    dirs = dirs[:len(dirs)/4]
    step = 25

    idx = np.arange(len(dirs))

    num_bin_file = int(np.ceil(len(dirs) / step))
    num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    num_val_bin_file = num_bin_file - num_train_bin_file

    # get training data
    train_dirs = []
    for i in range(num_train_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range(i * step, (i + 1) * step)]
        train_dirs.append(tmp_dir)

    # get val data
    val_dirs = []
    for i in range(num_val_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range((i + num_train_bin_file) * step, (i + 1 + num_train_bin_file) * step)]
        val_dirs.append(tmp_dir)

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)
    print(num_train_bin_file)
    print(num_val_bin_file)

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, 0, step, 'small_17tag_train_batch', i) for i in
        range(num_train_bin_file))

    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, num_train_bin_file, step, 'small_17tag_val_batch', i) for i in
        range(num_val_bin_file))


def processFile_parallel_alldir():
    dirs = glob(DATA_PATH + '/*')
    more_train_dirs = dirs[len(dirs)/4:]
    small_dirs = dirs[:len(dirs) / 4]
    step = 25

    idx = np.arange(len(dirs))
    val_idx = idx[int(len(dirs)/4 *TRAIN_RATIO) :len(dirs)/4]
    print(val_idx[0])
    print(val_idx[-1])
    train_idx = np.concatenate([ idx[:int(len(dirs)/4 *TRAIN_RATIO)], idx[len(dirs)/4:] ])
    num_bin_file = int(np.ceil(len(dirs) / step))
    num_val_bin_file = int(np.ceil(len(small_dirs)/step*(1-TRAIN_RATIO)))
    num_train_bin_file = num_bin_file - num_val_bin_file
    # print(train_idx.shape)
    # print(num_train_bin_file)
    # print(num_val_bin_file)

    # get training data
    train_dirs = []
    for i in range(num_train_bin_file):
        tmp_dir = [dirs[train_idx[j]] for j in range(i * step, (i + 1) * step)]
        train_dirs.append(tmp_dir)

    # get val data
    val_dirs = []
    for i in range(num_val_bin_file):
        tmp_dir = [dirs[val_idx[j]] for j in range((i ) * step, (i + 1 ) * step)]
        val_dirs.append(tmp_dir)

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)
    print(num_train_bin_file)
    print(num_val_bin_file)

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, 0, step, 'all_17tag_train_batch', i) for i in
        range(num_train_bin_file))

    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, num_train_bin_file, step, 'all_17tag_val_batch', i) for i in
        range(num_val_bin_file))



def main():
    # processFile_parallel()
    processFile_parallel_alldir()

if __name__ == "__main__":
    main()
