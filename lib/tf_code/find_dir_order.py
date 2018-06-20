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

DATA_PATH = '../data/SyntheticScatteringData3'
DATA_PATH_BIN = DATA_PATH + 'BIN_log_32'
ALL_TAG_FILE = '../data/17tags_meta.txt'
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


def read1stFile(dir, numdigits):
    file_names = os.listdir(dir)
    data_file = file_names[0]
    string_binary = ''
    pure_numbers = []
    # for data_file in file_names:
    if os.path.isfile(os.path.join(dir, data_file)):
        label, flag = parse_label(os.path.join(dir, data_file))
        if not flag:
            print(os.path.join(dir, data_file))
        else:
            label = label.astype('int16')
            label = list(label)
            pure_numbers +=label
            label_byte = struct.pack('h' * len(label), *label)
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
            pure_numbers +=image
            image_byte = struct.pack('h' * len(image), *image)
            string_binary += image_byte

    # return string_binary
    string_numbers = ''.join(str(e) for e in pure_numbers[0:numdigits])
    return string_numbers#[0:1000] #, string_binary


# compare the 1st file in all 500 directories, to see whether they are the same or different, if they are different, it would be easier
# from the experiment, we get the just compare the first 100000 numbers can distinguish those files
def compare1stFile():
    numdigits = 10000
    dirs = glob(DATA_PATH + '/*')
    firstFiles = getall1stFiles(dirs, numdigits)
    seen = set()
    unique = []
    for x in firstFiles:
        if x not in seen:
            unique.append(x)
            seen.add(x)
    print(len(seen))
    print(len(unique))
    # element = unique.pop()
    # print(element)

# get first files in all directories. The order of directory is got using glob. Only get first 10000 digits is enough
# os.listdir and glob have the same order. But for different locations, the order is different.
# original train_batch_0.batch were generated in /nfs/bigbang/boyu/Projects/Scattering. You can only reproduce the result in that location
def getall1stFiles(dirs, numdigits):
    # dirs = glob(DATA_PATH + '/*')
    print(len(dirs))
    # numdigits = 10000
    # read the first files in each dir
    num_cores = multiprocessing.cpu_count()
    firstFiles = Parallel(n_jobs=num_cores)(
        delayed(read1stFile)(dir, numdigits) for dir in dirs)

    return firstFiles

def getDirOrders_helper(bin_file_name, allFirstFiles, numdigits):
    # open and read the binary file
    files_perdir = 200
    num_dirs = 50
    size_perfile = (17 + 256*256) * 2
    currentFirstFiles = [None]*num_dirs
    with open(bin_file_name, 'rb') as f:
        x = f.read()
        print(len(x)/size_perfile)
        # only read the first files in each directory, only get the first 10000 numbers
        for idx in range(0, num_dirs):
            idx_instring_start = size_perfile*idx*files_perdir
            idx_instring_end = idx_instring_start + size_perfile
            tmp_file = struct.unpack('h'*(size_perfile/2), x[idx_instring_start: idx_instring_end])
            currentFirstFiles[idx] = ''.join(str(e) for e in tmp_file[0:numdigits])

    # find the index in allFirstFiles
    currentFileIdx = [None]*num_dirs
    for c_data_idx in range(0, num_dirs):
        c_data = currentFirstFiles[c_data_idx]
        for data_idx in range(0,len(allFirstFiles)) :
            data = allFirstFiles[data_idx]
            if c_data == data:
                currentFileIdx[c_data_idx] = data_idx
                break

    print(currentFileIdx)
    return currentFileIdx


# os.listdir and glob have the same order. But for different locations, the order is different.
# original train_batch_0.batch were generated in /nfs/bigbang/boyu/Projects/Scattering. You can only reproduce the result in that location
def main():
    numdigits = 10000
    dirs = glob(DATA_PATH + '/*')
    allFirstFiles = getall1stFiles(dirs, numdigits)
    bin_names = ['train_batch_%d.bin'%i for i in range(0, 8)]
    bin_names.extend(['val_batch_%d.bin'%i for i in range(0,2)])
    dirIdxes = [None]*len(bin_names)
    dirNames = [None]*len(bin_names)
    for i in range(0, len(bin_names)):
        idx_tmp = getDirOrders_helper(DATA_PATH_BIN+ '/'+ bin_names[i], allFirstFiles, numdigits)
        dirIdxes[i] = idx_tmp
        dirNames[i] = [ dirs[e] for e in idx_tmp]

    dirIdxes = np.array(dirIdxes)
    order_tosave = dict()
    order_tosave['dir_idxes'] = dirIdxes
    order_tosave['dir_names'] = dirNames

    sio.savemat('order_idx.mat', order_tosave)


if __name__ == "__main__":
    main()
