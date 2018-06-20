import glob
import numpy as np
import os
import scipy.io as sio
from scipy import misc

DATA_PATH = '../data/CHX_experimental_data/2016_03_13r/mini_image/'
DATA_NAME = glob.glob(DATA_PATH+'*.mat')
NUM_DATA = len(DATA_NAME)

IMAGE_SIZE = 224
if os.path.isfile('../data/new_realdata.mat'):
    data = sio.loadmat('../data/new_realdata.mat')
    DATA = data['images']
else:
    DATA = np.zeros([NUM_DATA, IMAGE_SIZE, IMAGE_SIZE, 1])

    for i in range(NUM_DATA):
        foo = sio.loadmat(DATA_NAME[i])
        foo = foo['detector_image']
        image = misc.imresize(foo, [IMAGE_SIZE, IMAGE_SIZE])
        # per image whitening
        image_mean = image.mean()
        image_std = image.std()
        image_adjusted_std = np.max([image_std, 1.0/np.sqrt(IMAGE_SIZE * IMAGE_SIZE)])
        whitened_image = (image - image_mean) / image_adjusted_std
        DATA[i,:,:,0] = whitened_image

    # save file
    data = dict()
    data['images'] = DATA
    sio.savemat('../data/new_realdata.mat', data)




