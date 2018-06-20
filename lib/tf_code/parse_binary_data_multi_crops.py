'''
the file parse the xml label file
and save both image and labels into binary files

'''

import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio
from glob import glob
import multiprocessing
from joblib import Parallel, delayed

DATA_PATH = '../data/SyntheticScatteringData_multi_crops2'
DATA_PATH_BIN = DATA_PATH + 'BIN_log_32'
ALL_TAG_FILE = '../data/17tags_meta.txt'
ALL_TAG_META = []
with open(ALL_TAG_FILE) as f:
    ALL_TAG_META = f.read().lower().split('\n')
NUM_TAGS = len(ALL_TAG_META)
# all tag_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
ALL_TAG_META = zip(np.arange(0, NUM_TAGS), ALL_TAG_META)
TRAIN_RATIO = 0.8
IMAGESIZE = 224

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

# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in float64 (double) format
def _binaryize_one_dir(dir):
    # file_names = os.listdir(dir)
    file_names = glob(dir + '/00*')
    string_binary = ''
    for data_file in file_names:
        if os.path.isfile(data_file):
            label, flag = parse_label(data_file)
            if not flag:
                print(os.path.join(data_file))
            else:
                label = label.astype('int16')
                label = list(label)
                label_byte = struct.pack('h'*len(label), *label)
                string_binary += label_byte

                # get the crop image file name
                crop_data_file = data_file.rsplit('/',1)
                crop_data_file = crop_data_file[0] + '/cropyx_' + crop_data_file[1]
                image = sio.loadmat(crop_data_file)
                # the shape of image is 224 * 224 * 3
                # image = image['image_crops']
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


# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in float64 (double) format
def _binaryize_one_dir_check(dir):
    # file_names = os.listdir(dir)
    file_names = glob(dir + '/00*')
    string_binary = ''
    for data_file in file_names:
        if os.path.isfile(data_file):
            label, flag = parse_label(data_file)
            if not flag:
                print(os.path.join(data_file))
            else:
                # label = label.astype('int16')
                # label = list(label)
                # label_byte = struct.pack('h'*len(label), *label)
                # string_binary += label_byte

                # get the crop image file name
                crop_data_file = data_file.rsplit('/',1)
                crop_data_file = crop_data_file[0] + '/crops_' + crop_data_file[1]
                image = sio.loadmat(crop_data_file)
                # the shape of image is 224 * 224 * 3
                image = image['image_crops']
                image = np.reshape(image, [-1])
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')
                if np.sum(image) == 0:
                    print(data_file)
                    print('all zero image')
                if np.any(np.isnan(image)):
                    print(data_file)
                    print('nan')


def processFile():
    dirs = os.listdir(DATA_PATH)
    step = 50
    idx = np.random.permutation(len(dirs))
    num_bin_file = int(np.ceil(len(dirs)/step))
    num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    num_val_bin_file = num_bin_file - num_train_bin_file

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)


    for i in range(num_train_bin_file):
        with open(os.path.join(DATA_PATH_BIN, 'train_batch_%d.bin' % i),'wb') as f:
            print('processing train_batch_%d.bin' % i)
            for j in range(step):
                dir_idx = int(idx[i*step + j])
                dir_name = os.path.join(DATA_PATH, dirs[dir_idx])
                # _binaryize_one_dir(dir_name)
                f.write(_binaryize_one_dir(dir_name))


    for i in range(num_val_bin_file):
        with open(os.path.join(DATA_PATH_BIN, 'val_batch_%d.bin' % i), 'wb') as f:
            print('processing val_batch_%d.bin' % i)
            for j in range(step):
                dir_idx = int(idx[ (i + num_train_bin_file) * step + j])
                dir_name = os.path.join(DATA_PATH, dirs[dir_idx])
                # _binaryize_one_dir(dir_name)
                f.write(_binaryize_one_dir(dir_name))


def _binaryize_one_list(all_dirs, idx_permutation, base_num, step_size,  save_name, i):
    tmp_dir = [all_dirs[idx_permutation[j]] for j in range((i+base_num)*step_size, (i+base_num+1)*step_size)]
    print('processing %s_%d.bin' % (save_name, i))
    with open(os.path.join(DATA_PATH_BIN, '%s_%d.bin' % (save_name, i)), 'wb') as f:
        for dir_list_i in tmp_dir:
            f.write(_binaryize_one_dir(dir_list_i))

def processFile_parrel():
    # dirs = os.listdir(DATA_PATH)
    dirs = glob(DATA_PATH+'/*')
    step = 50
    idx = np.random.permutation(len(dirs))
    num_bin_file = int(np.ceil(len(dirs) / step))
    num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    num_val_bin_file = num_bin_file - num_train_bin_file

    # get training data
    train_dirs = []
    for i in range(num_train_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range(i*step, (i+1)*step)]
        train_dirs.append(tmp_dir)

    # get val data
    val_dirs = []
    for i in range(num_val_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range((i+num_train_bin_file)*step,  (i+1+num_train_bin_file)*step)]
        val_dirs.append(tmp_dir)

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)
    print(num_train_bin_file)
    print(num_val_bin_file)

    num_cores = multiprocessing.cpu_count()-10
    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, 0, step,  'train_batch', i) for i in range(num_train_bin_file))

    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, num_train_bin_file, step, 'val_batch', i) for i in range(num_val_bin_file))

# check the input
def checkFile():
    # dirs = os.listdir(DATA_PATH)
    # step = 50
    # idx = np.random.permutation(len(dirs))
    # num_bin_file = int(np.ceil(len(dirs) / step))
    num_train_bin_file = 8
    num_val_bin_file = 2

    # if not os.path.exists(DATA_PATH_BIN):
    #     os.mkdir(DATA_PATH_BIN)

    for i in range(num_train_bin_file):
        with open(os.path.join(DATA_PATH_BIN, 'train_batch_%d.bin' % i), 'rb') as f:
            print('processing train_batch_%d.bin' % i)
            x = f.read()
            num_img_per_bin = len(x) / (17+224*224*3)/2
            for j in range(num_img_per_bin):
                label_j = struct.unpack('h'*17, x[(17+224*224*3)*2*j: (17+224*224*3)*2*j+17*2])
                image_j = struct.unpack('h'*224*224*3, x[(17+224*224*3)*2*j+17*2: (17+224*224*3)*2*(j+1)])
                if np.sum(label_j) == 0:
                    print('label all 0')
                    print(j)
                if np.sum(image_j) == 0:
                    print(j)
                    print('all zero image')
                if np.any(np.isnan(image_j)):
                    print(j)
                    print('nan')


    for i in range(num_train_bin_file):
        with open(os.path.join(DATA_PATH_BIN, 'val_batch_%d.bin' % i), 'rb') as f:
            print('processing val_batch_%d.bin' % i)
            x = f.read()
            num_img_per_bin = len(x) / (17+224*224*3)/2
            for j in range(num_img_per_bin):
                label_j = struct.unpack('h'*17, x[(17+224*224*3)*2*j: (17+224*224*3)*2*j+17*2])
                image_j = struct.unpack('h'*224*224*3, x[(17+224*224*3)*2*j+17*2: (17+224*224*3)*2*(j+1)])
                if np.sum(label_j) == 0:
                    print('label all 0')
                    print(j)
                if np.sum(image_j) == 0:
                    print(j)
                    print('all zero image')
                if np.any(np.isnan(image_j)):
                    print(j)
                    print('nan')

def main():
    # data_file_name = '../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat'
    # print(parse_label(data_file_name))
    processFile_parrel()

if __name__ == "__main__":
    main()
