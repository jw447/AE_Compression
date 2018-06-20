import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio

gt_dir_path = '../data/2016_03_13r_gt'
pred_dir_path = '../data/2016_03_13r_pred'

ALL_TAG_FILE = '../data/17tags_meta.txt'
ALL_TAG_META = []
with open(ALL_TAG_FILE) as f:
    ALL_TAG_META = f.read().lower().split('\n')
NUM_TAGS = len(ALL_TAG_META)
# all tag_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
ALL_TAG_META = zip(np.arange(0, NUM_TAGS), ALL_TAG_META)

tag_id = 4 # diffuse low q

# data_file_name could be like ../data/2016_03_13r_gt/f1165342-49a4-4dde-8a9b_2773_master.xml
# return a label vector of size [num_of_tags]
def parse_label(label_file_name):
    label_vector = 0

    if os.path.exists(label_file_name):
        root = ET.parse(label_file_name).getroot()
        for result in root[0]:
            attribute = result.attrib.get('name')
            attribute = attribute.rsplit('.', 1)
            # only care about high level tags
            attribute = attribute[1].split(':')[0]
            # check if that attribute is with the all_tag_file
            if ALL_TAG_META[tag_id][1] == attribute:
                label_vector = 1


    else:
        print ('%s does not exist!' %label_file_name)

    flag = bool(label_vector)
    return label_vector #, flag

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


def compute_accuracy():
    gt_files = os.listdir(gt_dir_path)
    pred_label = []
    gt_label = []
    for file_name in gt_files:
        if os.path.isfile(os.path.join(gt_dir_path, file_name)):
            gt_file_name = os.path.join(gt_dir_path, file_name)
            pred_file_name = os.path.join(pred_dir_path, file_name)
            gt_label.append(parse_label(gt_file_name))
            pred_label.append(parse_label(pred_file_name))

    labels = dict()
    labels['gt'] = gt_label
    labels['pred'] = pred_label

    sio.savemat('diff_lowq_labels.mat', labels)


def main():
    compute_accuracy()

if __name__ == "__main__":
    main()
