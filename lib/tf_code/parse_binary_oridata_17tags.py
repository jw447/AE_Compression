'''
the file parse the xml label file
and save both image and labels into binary files

'''

import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
from scipy import misc

DATA_PATH = '../data/ScatteringDataset'
DATA_PATH_BIN = DATA_PATH + 'BIN_log_32'
ALL_TAG_FILE = '../data/17tags_meta.txt'
ALL_TAG_META = []
with open(ALL_TAG_FILE) as f:
    ALL_TAG_META = f.read().lower().split('\n')
NUM_TAGS = len(ALL_TAG_META)
# all tag_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
ALL_TAG_META = zip(np.arange(0, NUM_TAGS), ALL_TAG_META)
IMAGESIZE = 256

# data_file_name could be like ../data/ScatteringDataset/data/2011Jan28-BrentCarey/direct_beam-1s_SAXS
# label file is lik ../data/ScatteringDataset/tags/2011Jan28-BrentCarey/direct_beam-1s_SAXS.tag
# return a label vector of size [num_of_tags]
def parse_label(data_file_name):
    data_path = data_file_name.rsplit('data/',1)
    label_file_name = data_path[0] + 'tags/' + data_path[1] + '.tag'
    label_vector = np.zeros([NUM_TAGS])
    exist = True
    if os.path.exists(label_file_name):
        root = ET.parse(label_file_name).getroot()
        # get tags
        for tags in root.iter('tag'):
            tags_name = tags.text
            # only care about high level tags
            attribute = tags_name.split(':')[0].lower()
            # check if that attribute is with the all_tag_file
            for i in range(NUM_TAGS):
                if ALL_TAG_META[i][1] == attribute:
                    label_vector[ALL_TAG_META[i][0]] = 1
    else:
        # print ('%s does not exist!' % label_file_name)
        exist = False

    flag = bool(sum(label_vector))
    return label_vector, flag, exist

# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in int16 format
# take the log first, then resize to 256*256
def _binaryize_one_dir(dir):
    file_names = os.listdir(dir)
    string_binary = ''
    cnt = 0
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag, exist = parse_label(os.path.join(dir, data_file))
            if not flag and exist:
                print(os.path.join(dir, data_file) + 'does not contain tags you are looking for')
            elif exist:
                cnt += 1
                label = label.astype('int16')
                label = list(label)
                label_byte = struct.pack('h'*len(label), *label)    # int16 encoding
                string_binary += label_byte
                image = misc.imread(os.path.join(dir, data_file))
                # resize to 256*256
                image_resize = misc.imresize(image, [256,256])
                image_resize = image_resize.astype('float')
                # resize will change the max and min value in a picture to a value in 0-255, map it back
                image = image_resize * image.max() / image_resize.max()
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')

                image = list(image)
                image_byte = struct.pack('h'*len(image), *image)
                string_binary += image_byte
        # if there are subdirectories in the dir
        elif os.path.isdir(os.path.join(dir, data_file)) and data_file[0] != '.':
            print('writing %s' % os.path.join(dir, data_file))
            string_binary += _binaryize_one_dir(os.path.join(dir, data_file))

    print('%d images in %s' % (cnt, dir) )
    return string_binary

# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in int16 format
# resize to 256*256, then take the log
def _binaryize_one_dir2(dir):
    file_names = os.listdir(dir)
    string_binary = ''
    cnt = 0
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag, exist = parse_label(os.path.join(dir, data_file))
            if not flag and exist:
                print(os.path.join(dir, data_file) + 'does not contain tags you are looking for')
            elif exist:
                cnt += 1
                label = label.astype('int16')
                label = list(label)
                label_byte = struct.pack('h'*len(label), *label)    # int16 encoding
                string_binary += label_byte
                image = misc.imread(os.path.join(dir, data_file))
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')
                # the shape of image is 256*256
                image = misc.imresize(image, [256,256])
                image = np.reshape(image, [-1])
                image = list(image)
                image_byte = struct.pack('h'*len(image), *image)
                string_binary += image_byte
        # if there are subdirectories in the dir
        elif os.path.isdir(os.path.join(dir, data_file)) and data_file[0] != '.':
            print('writing %s' % os.path.join(dir, data_file))
            string_binary += _binaryize_one_dir2(os.path.join(dir, data_file))

    print('%d images in %s' % (cnt, dir) )
    return string_binary


# helper function to get the label of one directionary
def _getLabel_one_dir(dir):
    file_names = os.listdir(dir)
    cnt = 0
    label_all = []
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag, exist = parse_label(os.path.join(dir, data_file))
            if not flag and exist:
                print(os.path.join(dir, data_file) + 'does not contain tags you are looking for')
            elif exist:
                cnt += 1
                label = label.astype('int16')
                label_all.append(label)
        # if there are subdirectories in the dir
        elif os.path.isdir(os.path.join(dir, data_file)) and data_file[0] != '.':
            print('writing %s' % os.path.join(dir, data_file))
            label_all.extend(_getLabel_one_dir(os.path.join(dir, data_file)))

    print('%d images in %s' % (cnt, dir))
    return label_all

def getAllLabel():
    dirs = os.listdir(os.path.join(DATA_PATH, 'data'))
    label_all = []
    for dir_idx in range(len(dirs)):
        if os.path.isdir(os.path.join(DATA_PATH, 'data', dirs[dir_idx])):
            dir_name = os.path.join(DATA_PATH, 'data', dirs[dir_idx])
            print('processing %s' % dir_name)
            label_all.extend(_getLabel_one_dir(dir_name))
    return label_all


def processFile():
    dirs = os.listdir(os.path.join(DATA_PATH, 'data'))

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)

    with open(os.path.join(DATA_PATH_BIN, 'ori_resize_log_batch.bin'), 'wb') as f:
        print('processing ori_batch.bin')
        for dir_idx in range(len(dirs)):
            if os.path.isdir(os.path.join(DATA_PATH, 'data', dirs[dir_idx])):
                dir_name = os.path.join(DATA_PATH, 'data', dirs[dir_idx])
                print('writing %s' % dir_name)
                f.write(_binaryize_one_dir2(dir_name))




# def main():
#     processFile()
#
# if __name__ == "__main__":
#     main()
