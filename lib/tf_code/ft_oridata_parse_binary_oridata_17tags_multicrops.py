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
import scipy.io as sio
import multiprocessing
from joblib import Parallel, delayed

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
# CROP_SCALES = [256, 384, 448, 512, 640, 768, 896]   # 95
# CROP_SCALES = [256, 288, 320, 384, 416, 448, 480, 512] # 28 crops
# CROP_SCALES = [256, 320, 384, 448, 512]  # 19
# CROP_SCALES = [256, 320, 384, 512]  # 11
CROP_SCALES = [256, 384, 512]  # 35, step 64, instead of 128
# CROP_SCALES = [256, 384, 512, 768]  # 39
start_idx = []
NUM_CROPS = 0
for scales_idx in range(len(CROP_SCALES)):
    start_idx.append(np.arange(0, CROP_SCALES[scales_idx]-256+1, 64))
    # start_idx.append(np.arange(0, CROP_SCALES[scales_idx] - 256 + 1, 128))
    NUM_CROPS = NUM_CROPS + len(start_idx[scales_idx]) * len(start_idx[scales_idx])
print('num crops: %d' %NUM_CROPS)
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

                image = misc.imread(os.path.join(dir, data_file))
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')
                for scale_idx in range(len(CROP_SCALES)):
                    scale_i = CROP_SCALES[scale_idx]
                    # resize image to that scale
                    image_resize = misc.imresize(image, [scale_i, scale_i])

                    # do crop
                    img_crops = []
                    for axis_x in range(len(start_idx[scale_idx])):
                        for axis_y in range(len(start_idx[scale_idx])):
                            # print(start_idx[scale_idx][axis_x])
                            # print(start_idx[scale_idx][axis_y])
                            # print(scale_idx)
                            tmp_img = image_resize[start_idx[scale_idx][axis_x]:start_idx[scale_idx][axis_x]+256, start_idx[scale_idx][axis_y]:start_idx[scale_idx][axis_y]+256]
                            img_crops.append(tmp_img)
                            # print(tmp_img.shape)
                            assert tmp_img.shape == (256 ,256)
                    for crops_idx in range(len(img_crops)):
                        image_crop = np.reshape(img_crops[crops_idx], [-1])
                        image_crop = list(image_crop)
                        assert len(image_crop) == 256 * 256
                        image_byte = struct.pack('h' * len(image_crop), *image_crop)
                        string_binary += label_byte
                        string_binary += image_byte
                    # if (NUM_CROPS_PER_SCALE[scale_idx]) == 1:
                    #     image_towrite = np.reshape(image_resize, [-1])
                    #     image_towrite = list(image_towrite)
                    #     image_byte = struct.pack('h'*len(image_towrite), *image_towrite)
                    #     string_binary += label_byte
                    #     string_binary += image_byte
                    # # do 5 crops
                    # else:
                    #     img_crops = []
                    #     img_crops.append( image_resize[:256,:256])
                    #     img_crops.append( image_resize[-256:,-256:])
                    #     img_crops.append(image_resize[:256, -256:])
                    #     img_crops.append(image_resize[-256:, :256])
                    #     img_crops.append( image_resize[scale_i/2-128:scale_i/2+128, scale_i/2-128:scale_i/2+128 ])
                    #     for crops_idx in range(NUM_CROPS_PER_SCALE[scale_idx]):
                    #         image_crop = np.reshape(img_crops[crops_idx], [-1])
                    #         image_crop = list(image_crop)
                    #         assert len(image_crop) == 256*256
                    #         image_byte = struct.pack('h' * len(image_crop), *image_crop)
                    #         string_binary += label_byte
                    #         string_binary += image_byte

        # if there are subdirectories in the dir
        elif os.path.isdir(os.path.join(dir, data_file)) and data_file[0] != '.':
            print('writing %s' % os.path.join(dir, data_file))
            string_binary += _binaryize_one_dir2(os.path.join(dir, data_file))

    print('%d images in %s' % (cnt, dir) )
    return string_binary



# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in int16 format
# resize to 256*256, then take the log
def _binaryize_one_dir2_parfor(dir):
    file_names = os.listdir(dir)
    # cnt = len(file_names)
    num_cores = multiprocessing.cpu_count()
    string_binary = Parallel(n_jobs=num_cores)(delayed(_binaryize_one_dir2_one_file)(dir, data_file)
        for data_file in file_names)
    string_binary = ''.join(string_binary)
    cnt = len(string_binary) / (NUM_CROPS*(17+256*256)*2)
    print('%d images in %s' % (cnt, dir))
    return string_binary

def _binaryize_one_dir2_one_file(dir, data_file):
    string_binary = ''
    if os.path.isfile(os.path.join(dir, data_file)):
        label, flag, exist = parse_label(os.path.join(dir, data_file))
        if not flag and exist:
            print(os.path.join(dir, data_file) + 'does not contain tags you are looking for')
        elif exist:
            label = label.astype('int16')
            label = list(label)
            label_byte = struct.pack('h'*len(label), *label)    # int16 encoding

            image = misc.imread(os.path.join(dir, data_file))
            # take the log
            image = np.log(image) / np.log(1.0414)
            image[np.isinf(image)] = 0
            image = image.astype('int16')
            for scale_idx in range(len(CROP_SCALES)):
                scale_i = CROP_SCALES[scale_idx]
                # resize image to that scale
                image_resize = misc.imresize(image, [scale_i, scale_i])

                # do crop
                img_crops = []
                for axis_x in range(len(start_idx[scale_idx])):
                    for axis_y in range(len(start_idx[scale_idx])):
                        # print(start_idx[scale_idx][axis_x])
                        # print(start_idx[scale_idx][axis_y])
                        # print(scale_idx)
                        tmp_img = image_resize[start_idx[scale_idx][axis_x]:start_idx[scale_idx][axis_x]+256, start_idx[scale_idx][axis_y]:start_idx[scale_idx][axis_y]+256]
                        img_crops.append(tmp_img)
                        # print(tmp_img.shape)
                        assert tmp_img.shape == (256 ,256)
                for crops_idx in range(len(img_crops)):
                    image_crop = np.reshape(img_crops[crops_idx], [-1])
                    image_crop = list(image_crop)
                    assert len(image_crop) == 256 * 256
                    image_byte = struct.pack('h' * len(image_crop), *image_crop)
                    string_binary += label_byte
                    string_binary += image_byte
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

def _get_file_names():
    dirs = os.listdir(os.path.join(DATA_PATH, 'data'))
    # get all directories and sub directories
    alldirs = []

    for dir in dirs:
        if os.path.isdir(os.path.join(DATA_PATH, 'data', dir)):
            subdir = glob(os.path.join(DATA_PATH, 'data', dir) + '/*/')
            if len(subdir) == 0:
                alldirs.append(os.path.join(DATA_PATH, 'data', dir))
            else:
                alldirs.extend(subdir)

    all_file_names = []
    all_labels = []
    for dir in alldirs:
        files_in_dir = []
        label_in_dir = []
        file_names = os.listdir(dir)
        for data_file in file_names:
            if os.path.isfile(os.path.join(dir, data_file)):
                label, flag, exist = parse_label(os.path.join(dir, data_file))
                if flag:
                    files_in_dir.append(str(os.path.join(dir, data_file)))
                    label_in_dir.append(label)
        all_file_names.append(files_in_dir)
        all_labels.append(label_in_dir)

    file_name_dataset = dict()
    file_name_dataset['file_name'] = all_file_names
    label_dataset = dict()
    label_dataset['label'] = all_labels
    sio.savemat('../data/ScatteringDataset/ori_binary_label.mat', label_dataset)




def processFile():
    dirs = os.listdir(os.path.join(DATA_PATH, 'data'))

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)

    # get all directories and sub directories
    alldirs = []
    for dir in dirs:
        if os.path.isdir(os.path.join(DATA_PATH, 'data', dir)):
            subdir = glob(os.path.join(DATA_PATH, 'data', dir)+'/*/')
            if len(subdir) == 0:
                alldirs.append(os.path.join(DATA_PATH, 'data', dir))
            else:
                alldirs.extend(subdir)

    num_cores = multiprocessing.cpu_count()
    files_dir = Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_dir2)(alldirs[i])
        for i in range(len(alldirs)))

    # files_dir = []
    # for i in range(len(alldirs)):
    #     files_dir.append(_binaryize_one_dir2(alldirs[i]))

    for i in range(len(alldirs)):
        # create the leave i out data, for training and for testing
        # with open(os.path.join(DATA_PATH_BIN, '16crops_ori_resize_log_lo-%d_train.bin' % i), 'wb') as f:
        #     print('processing 16crops_ori_resize_log_lo-%d_train.bin' % i)
        #     for dir_idx in range(len(alldirs)):
        #         if dir_idx != i:
        #             f.write(files_dir[dir_idx])

        with open(os.path.join(DATA_PATH_BIN, str(NUM_CROPS)+'crops_ori_resize_log_lo-%d_test.bin' % i), 'wb') as f:
            print('processing '+ str(NUM_CROPS)+'crops_ori_resize_log_lo-%d_test.bin' % (i))
            f.write(files_dir[i])



def processFile_parfor(dir_idx):
    dirs = os.listdir(os.path.join(DATA_PATH, 'data'))

    if not os.path.exists(DATA_PATH_BIN):
        os.mkdir(DATA_PATH_BIN)

    # get all directories and sub directories
    alldirs = []
    for dir in dirs:
        if os.path.isdir(os.path.join(DATA_PATH, 'data', dir)):
            subdir = glob(os.path.join(DATA_PATH, 'data', dir) + '/*/')
            if len(subdir) == 0:
                alldirs.append(os.path.join(DATA_PATH, 'data', dir))
            else:
                alldirs.extend(subdir)
    # files_dir = []
    # for i in range(len(alldirs)):
    #     files_dir.append(_binaryize_one_dir2(alldirs[i]))

    # for i in range(len(alldirs)):
    i = dir_idx
        # create the leave i out data, for training and for testing
        # with open(os.path.join(DATA_PATH_BIN, '16crops_ori_resize_log_lo-%d_train.bin' % i), 'wb') as f:
        #     print('processing 16crops_ori_resize_log_lo-%d_train.bin' % i)
        #     for dir_idx in range(len(alldirs)):
        #         if dir_idx != i:
        #             f.write(files_dir[dir_idx])

    string_binary = _binaryize_one_dir2_parfor(alldirs[i])
    with open(os.path.join(DATA_PATH_BIN, str(NUM_CROPS) + 'crops_ori_resize_log_lo-%d_test.bin' % i),
              'wb') as f:
        print('processing ' + str(NUM_CROPS) + 'crops_ori_resize_log_lo-%d_test.bin' % (i))
        f.write(string_binary)


# def main():
#     processFile()
#
# if __name__ == "__main__":
#     main()
