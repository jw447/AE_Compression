# translate Get_Patches.m to python
# implement efficient sliding window

import numpy as np
import scipy

def sub2ind(array_shape, rows, cols):
    return cols*array_shape[0] + rows


def Get_Patches(imreshape, PATCH_SIZE, STEP):
    # get number of dimensions for image (for grayscale, num_dims = 2)
    # NUM_DIMS = im.ndim
    # assert NUM_DIMS == 2
    im_size = np.sqrt(imreshape.shape[0]).astype('int')
    NUM_PATCHES_ONE_DIM = (im_size - PATCH_SIZE) / STEP + 1

    # imreshape = im.transpose().reshape([-1])
    start_idxs = np.arange(0, im_size-PATCH_SIZE+1, STEP)
    startCmbn_0 = np.tile(start_idxs, [1,NUM_PATCHES_ONE_DIM])[0]
    startCmbn_1 = np.tile(start_idxs, [NUM_PATCHES_ONE_DIM,1]).transpose().reshape([-1])

    pix_idx = np.arange(0, PATCH_SIZE)
    pixCmbn_0 = np.tile(pix_idx, [1,PATCH_SIZE])[0]
    pixCmbn_1 = np.tile(pix_idx, [PATCH_SIZE, 1]).transpose().reshape([-1])

    startLinIdxs = sub2ind([im_size, im_size], startCmbn_0, startCmbn_1)
    winLinIdxs = sub2ind([im_size, im_size], pixCmbn_0, pixCmbn_1)

    start_locs = np.array([startCmbn_0, startCmbn_1])

    A = np.tile(winLinIdxs, [len(startLinIdxs), 1] ).transpose() + np.tile(startLinIdxs, [len(winLinIdxs), 1])
    D = imreshape[A]

    return D, start_locs

