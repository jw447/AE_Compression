'''
utility functions to save and load hdf5 files
the size of the each data element is exactly as what it is in matlab
'''

import h5py
import numpy as np


# cData is a list, each element is a np array,
# the size of this array is the same as it it in matlab
def load(dataFile):
    with h5py.File(dataFile, 'r') as hf:
        dims = hf.get('/meta')
        dims = int(np.prod(dims))
        cData = []
        for i in range(dims):
            tmp = np.array(hf.get('/'+str(i+1)))
            # the size of tmp is the inverse of what it is in matlab, need to transpose
            tmp_ndim = tmp.ndim
            tmp = np.transpose(tmp, np.arange(tmp_ndim-1, -1, -1))
            cData.append(tmp)
    return cData

# cData is a list
# dims is a n*1 np array
# the size of each data element is the same as it is in matlab
def save(fileName, cData, dims):
    with h5py.File(fileName, 'w') as hf:
        n = np.prod(dims).astype(int)
        if n != len(cData):
            print('Error: the number of elements in cData must match dims')
            print('#cData: ' + str(len(cData)) + ', dims: '+str(n))
            return

        hf.create_dataset('/meta', data=dims)
        for i in range(n):
            tmp = cData[i]
            # the size of tmp is the inverse of what it is in matlab, need to transpose
            tmp_ndim = tmp.ndim
            tmp = np.transpose(tmp, np.arange(tmp_ndim - 1, -1, -1))
            hf.create_dataset(str(i+1), data=tmp)




