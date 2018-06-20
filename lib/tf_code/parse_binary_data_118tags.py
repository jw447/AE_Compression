'''
the file parse the xml label file
and save both image and labels into binary files

'''

import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio

DATA_PATH = '../data/SyntheticScatteringData3'
DATA_PATH_BIN = DATA_PATH + 'BIN_log_32'
ALL_TAG_FILE = '../data/118tags_meta.txt'
ALL_TAG_META = []
with open(ALL_TAG_FILE) as f:
    ALL_TAG_META = f.read().lower().split('\n')
NUM_TAGS = len(ALL_TAG_META)
# all tag_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
ALL_TAG_META = zip(np.arange(0, NUM_TAGS), ALL_TAG_META)
TRAIN_RATIO = 0.8
IMAGESIZE = 256

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
            attribute = result.attrib.get('name').lower()
            attribute = attribute.rsplit('.', 1)
            # only care about high level tags
            high_level_tag = attribute[1].split(':')[0]
            detail_tag = attribute[1]
            # check if that attribute is with the all_tag_file
            for i in range(NUM_TAGS):
                if ALL_TAG_META[i][1] == high_level_tag:
                    label_vector[ALL_TAG_META[i][0]] = 1
                if ALL_TAG_META[i][1] == detail_tag:
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
    file_names = os.listdir(dir)
    string_binary = ''
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag = parse_label(os.path.join(dir, data_file))
            if not flag:
                print(os.path.join(dir, data_file))
            else:
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

# get the label for one directory
def _get_label_one_dir(dir):
    file_names = os.listdir(dir)
    all_labels = []
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag = parse_label(os.path.join(dir, data_file))
            if not flag:
                print(os.path.join(dir, data_file))
            else:
                all_labels.append(label)

    return all_labels



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
        with open(os.path.join(DATA_PATH_BIN, 'train_batch_118tags_%d.bin' % i),'wb') as f:
            print('processing train_batch_118tags_%d.bin' % i)
            for j in range(step):
                dir_idx = int(idx[i*step + j])
                dir_name = os.path.join(DATA_PATH, dirs[dir_idx])
                # _binaryize_one_dir(dir_name)
                f.write(_binaryize_one_dir(dir_name))


    for i in range(num_val_bin_file):
        with open(os.path.join(DATA_PATH_BIN, 'val_batch_118tags_%d.bin' % i), 'wb') as f:
            print('processing val_batch_118tags_%d.bin' % i)
            for j in range(step):
                dir_idx = int(idx[ (i + num_train_bin_file) * step + j])
                dir_name = os.path.join(DATA_PATH, dirs[dir_idx])
                # _binaryize_one_dir(dir_name)
                f.write(_binaryize_one_dir(dir_name))


# get the label distribution matrix
# the distribution matrix is a matrix of size [num_data, num_label]
# the entry of 1 represent data_i has label_j
def getLabelDis():
    dirs = os.listdir(DATA_PATH)
    step = 50
    idx = np.random.permutation(len(dirs))
    num_bin_file = int(np.ceil(len(dirs) / step))
    num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    num_val_bin_file = num_bin_file - num_train_bin_file

    train_labels = []
    for i in range(num_train_bin_file):
        print('processing train_batch_118tags_%d.bin' % i)
        for j in range(step):
            dir_idx = int(idx[i*step + j])
            dir_name = os.path.join(DATA_PATH, dirs[dir_idx])
            train_labels.extend(_get_label_one_dir(dir_name))


    val_labels = []
    for i in range(num_val_bin_file):
        print('processing val_batch_118tags_%d.bin' % i)
        for j in range(step):
            dir_idx = int(idx[ (i + num_train_bin_file) * step + j])
            dir_name = os.path.join(DATA_PATH, dirs[dir_idx])
            val_labels.extend(_get_label_one_dir(dir_name))

    label_dataset = dict()
    label_dataset['train_labels'] = train_labels
    label_dataset['val_labels'] = val_labels
    sio.savemat(os.path.join(DATA_PATH_BIN, 'label_dis.mat'), label_dataset)
# def main():
#     data_file_name = '../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat'
#     print(parse_label(data_file_name))
#
# if __name__ == "__main__":
#     main()
