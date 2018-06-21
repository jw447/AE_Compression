'''
the file parse the xml label file
and save both image and labels into binary files

'''

import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio
import multiprocessing
from joblib import Parallel, delayed
from glob import glob


DATA_PATH = '../data/SyntheticScatteringData3'
DATA_PATH_BIN = DATA_PATH + 'BIN_log_32'
# ALL_TAG_META = ['linear beamstop']
ALL_TAG_META = ['polycrystalline']
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
def _binaryize_one_list(train_names, train_labels, save_name, i):
    string_binary = ''
    print('processing %s_%d.bin' % (save_name, i))
    name_list_i = train_names[i * 10000: min((i + 1) * 10000, len(train_names))]
    label_list_i = train_labels[i * 10000: min((i + 1) * 10000, len(train_names))]
    with open('%s_%d.bin' %(save_name,i), 'wb') as f:
        for idx in range(len(name_list_i)):
            label = [label_list_i[idx]]
            # label = list(label)
            label_byte = struct.pack('h'*len(label), *label)
            string_binary += label_byte
            image = sio.loadmat(name_list_i[idx])
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
            f.write(label_byte)
            f.write(image_byte)
    # return string_binary


# helper function to get the label of one directionary
def _getLabel_one_dir(dir):
    # file_names = os.listdir(dir)
    file_names = glob(dir + '/00*')
    cnt = 0
    label_all = []
    with_file_names = []
    wo_file_names = []
    for data_file in file_names:
        if os.path.isfile(data_file):
            label, flag = parse_label(data_file)
            if not flag:
                wo_file_names.append(data_file)
                # print (data_file + ' does not contain tags you are looking for')
            else:
                with_file_names.append(data_file)
                # cnt += 1
                # label = label.astype('int16')
                # label_all.append(label)
    names = []
    names.append(with_file_names)
    names.append(wo_file_names)
    names.append(dir)
    return names


def getAllLabel():
    dirs = os.listdir(DATA_PATH)
    # label_all = [None] * len(dirs)
    num_cores = multiprocessing.cpu_count()
    names = Parallel(n_jobs=num_cores/2)(delayed(_getLabel_one_dir)(os.path.join(DATA_PATH,dir_names)) for dir_names in dirs)
    # for dir_name in dirs:
    #     if os.path.isdir(os.path.join(DATA_PATH, 'data', dirs[dir_idx])):
    #         dir_name = os.path.join(DATA_PATH, 'data', dirs[dir_idx])
    #         print('processing %s' % dir_name)
    #         label_all.extend(_getLabel_one_dir(dir_name))

    # get positive names
    pos_names = []
    neg_names = []
    for i in range(len(names)):
        if len(names[i][0]) > 0:
            pos_names.extend(names[i][0])
        if len(names[i][1]) > 0:
            neg_names.extend(names[i][1])

    return pos_names, neg_names



def processFile():

    pos_names, neg_names = getAllLabel()
    num_pos = len(pos_names)
    print('num_positive images: %d' % num_pos)
    num_pos_train = int(num_pos * TRAIN_RATIO)
    print('num_positive images train: %d' % num_pos_train)

    pos_train_names = pos_names[:num_pos_train]
    pos_val_names = pos_names[num_pos_train:]
    idx = np.random.permutation(num_pos)
    neg_train_names = [neg_names[i] for i in idx[:num_pos_train]]
    neg_val_names = [neg_names[i] for i in idx[num_pos_train:]]

    # get training data
    print(np.ceil(num_pos_train * 2 / 10000.0))
    num_train_bin_file = int(np.ceil(num_pos_train * 2 / 10000.0))
    train_names = []
    train_labels = []
    for i in range(num_pos_train):
        train_names.append(pos_train_names[i])
        train_labels.append(1)
        train_names.append(neg_train_names[i])
        train_labels.append(0)

    # get val data
    print((num_pos - num_pos_train) * 2 / 10000.0)
    num_val_bin_file = int(np.ceil((num_pos - num_pos_train) * 2 / 10000.0))
    val_names = []
    val_labels = []
    for i in range(num_pos - num_pos_train):
        val_names.append(pos_val_names[i])
        val_names.append(neg_val_names[i])
        val_labels.append(1)
        val_labels.append(0)

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)
    print(num_train_bin_file)
    print(num_val_bin_file)

    num_cores = multiprocessing.cpu_count()
    # Parallel(n_jobs=num_cores )(delayed(_binaryize_one_list)(train_names, train_labels, 'n_linearbs_train_batch', i) for i in range(num_train_bin_file))
    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(train_names, train_labels, os.path.join(DATA_PATH_BIN, 'n_polycrl_train_batch') , i) for i in
        range(num_train_bin_file))
    # for i in range(num_train_bin_file):
    #     with open(os.path.join(DATA_PATH_BIN, 'linearbs_train_batch_%d.bin' % i),'wb') as f:
    #         print('processing linearbs_train_batch_%d.bin' % i)
    #         name_list_i = train_names[i*10000: min((i+1)*10000, len(train_names))]
    #         label_list_i = train_labels[i*10000: min((i+1)*10000, len(train_names))]
    #         f.write(_binaryize_one_list(name_list_i, label_list_i))

    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(val_names, val_labels, os.path.join(DATA_PATH_BIN, 'n_polycrl_val_batch'), i) for i in range(num_val_bin_file))
    # Parallel(n_jobs=num_cores)(
    #     delayed(_binaryize_one_list)(val_names, val_labels, 'n_linearbs_val_batch', i) for i in range(num_val_bin_file))
    # for i in range(num_val_bin_file):
    #     with open(os.path.join(DATA_PATH_BIN, 'linearbs_val_batch_%d.bin' % i), 'wb') as f:
    #         print('processing linearbs_val_batch_%d.bin' % i)
    #         name_list_i = val_names[i * 10000: min((i + 1) * 10000, len(val_names))]
    #         label_list_i = val_labels[i * 10000: min((i + 1) * 10000, len(val_names))]
    #         f.write(_binaryize_one_list(name_list_i, label_list_i))



def main():
    # data_file_name = '../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat'
    processFile()
    # print(parse_label(data_file_name))

if __name__ == "__main__":
    main()
