'''
the file parse the xml label file
and save both image and labels into binary files

'''

import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio
from scipy.ndimage.filters import convolve as im_conv2
from scipy.ndimage.filters import gaussian_filter as im_gaussian
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
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
gaussian_std = 2
thresh = 10

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
                label = label.astype('int32')
                label = list(label)
                label_byte = struct.pack('f'*len(label), *label)
                string_binary += label_byte
                image = sio.loadmat(os.path.join(dir, data_file))
                # the shape of image is 256*256
                image = image['detector_image']
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0

                # gaussian blur
                image = im_gaussian(image, gaussian_std)
                # compute gradient
                grad_x = im_conv2(image, sobel_x)
                grad_y = im_conv2(image, sobel_y)

                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                grad_x = grad_x / (grad_mag + np.finfo(float).eps)
                grad_y = grad_y / (grad_mag + np.finfo(float).eps)
                if np.sum(np.abs(grad_x)) == 0:
                    print('1 error x')
                    print('%s, %s'%(dir, data_file))
                if np.sum(np.abs(grad_y)) == 0:
                    print('1 error y')
                    print('%s, %s' % (dir, data_file))

                grad_x_th = grad_x.copy()
                grad_y_th = grad_y.copy()
                grad_x_th[grad_mag<thresh] = 0
                grad_y_th[grad_mag < thresh] = 0
                # grad_mag = grad_mag / np.max(grad_mag)


                # grad_mag_sorted = np.sort(np.reshape(grad_mag, [-1]))
                # thre_low = grad_mag_sorted[int(256*256*0.05)]
                # thre_high = grad_mag_sorted[int(256*256*0.95)]
                # # print(np.max(grad_mag))
                # grad_x = grad_x / (grad_mag + np.finfo(float).eps)
                # grad_y = grad_y / (grad_mag + np.finfo(float).eps)
                #
                # grad_x[grad_mag < thre_low] = 0
                # grad_y[grad_mag < thre_low] = 0
                #
                # grad_mag[grad_mag < thre_low] = 0
                # grad_mag[grad_mag > thre_high] = thre_high

                # grad_mag = grad_mag / thre_high


                if np.sum(np.abs(grad_x_th)) > 0.5 or np.sum(np.abs(grad_y_th)) > 0.5:
                    grad_x = grad_x_th
                    grad_y = grad_x_th

                grad_mag = grad_mag / np.max(grad_mag)

                if np.sum(np.abs(grad_x)) == 0:
                    print('error x')
                    print('%s, %s'%(dir, data_file))
                if np.sum(np.abs(grad_y)) == 0:
                    print('error y')
                    print('%s, %s' % (dir, data_file))
                grad_x = np.reshape(grad_x, [-1])
                grad_y = np.reshape(grad_y, [-1])
                grad_mag = np.reshape(grad_mag, [-1])
                grad_x = grad_x.astype('float32')
                grad_y = grad_y.astype('float32')
                grad_mag = grad_mag.astype('float32')

                grad_x = list(grad_x)
                grad_y = list(grad_y)
                grad_mag = list(grad_mag)

                grad_x_byte = struct.pack('f'*len(grad_x), *grad_x)
                grad_y_byte = struct.pack('f' * len(grad_y), *grad_y)
                grad_mag_byte = struct.pack('f' * len(grad_mag), *grad_mag)
                string_binary += grad_x_byte
                string_binary += grad_y_byte
                string_binary += grad_mag_byte

    return string_binary


def _binaryize_one_list(all_dirs, idx_permutation, base_num, step_size,  save_name, i):
    tmp_dir = [all_dirs[idx_permutation[j]] for j in range((i+base_num)*step_size, (i+base_num+1)*step_size)]
    print('processing %s_%d.bin' % (save_name, i))
    with open(os.path.join(DATA_PATH_BIN, '%s_%d.bin' % (save_name, i)), 'wb') as f:
        for dir_list_i in tmp_dir:
            f.write(_binaryize_one_dir(dir_list_i))


def processFile():
    # dirs = os.listdir(DATA_PATH)
    dirs = glob(DATA_PATH + '/*')
    step = 50
    idx = np.random.permutation(len(dirs))
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
        delayed(_binaryize_one_list)(dirs, idx, 0, step, 'grad_thre-10_train_batch', i) for i in range(num_train_bin_file))

    Parallel(n_jobs=num_cores)(
        delayed(_binaryize_one_list)(dirs, idx, num_train_bin_file, step, 'grad_thre-10_val_batch', i) for i in
        range(num_val_bin_file))




# def main():
#     data_file_name = '../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat'
#     print(parse_label(data_file_name))
#
# if __name__ == "__main__":
#     main()
