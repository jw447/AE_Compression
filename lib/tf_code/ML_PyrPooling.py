# translate from ML_PyrPooling.m

import numpy as np
import scipy.io as sio

def pool(FeatLocs, FeatVecs, poolArea, poolMethod):
    [k, n] = FeatLocs.shape
    A = np.concatenate( [FeatLocs >= np.tile(poolArea[:,0].reshape([k,1]), [1, n]), FeatLocs <= np.tile(poolArea[:,1].reshape([k,1]), [1, n])], 0)
    inPoolAreaIdxs = np.all(A, 0)
    nFeatInArea = sum(inPoolAreaIdxs)
    if nFeatInArea > 0:
        if poolMethod == 'mean':
            poolVec = np.mean(FeatVecs[:,inPoolAreaIdxs], 1)
        elif poolMethod == 'max':
            poolVec = np.max(FeatVecs[:, inPoolAreaIdxs], 1)
        elif poolMethod == 'min':
            poolVec = np.min(FeatVecs[:, inPoolAreaIdxs], 1)
        elif poolMethod == 'sum':
            poolVec = np.sum(FeatVecs[:, inPoolAreaIdxs], 1)
    else:
        print('no feature in this area')

    return poolVec


# def pyrPoolhelper(FeatLocs, FeatVecs, poolArea, poolMethod, levelWs):


def pyrPool(FeatLocs, FeatVecs, poolArea, poolMethod, levelWs ):
    nLevel = levelWs.shape[0]
    [k, d] = FeatLocs.shape

    midPnts = np.mean(poolArea,1)
    # poolAreas = np.array([poolArea[:,0], midPnts, midPnts+ np.finfo(float).eps, poolArea[:,1]]).transpose()

    poolVec = []
    # nF = np.zeros([1, 2**k])

    # start from fine grained level
    nDiv = 2**(nLevel-1)
    poolVec_tmp = []
    for div_i in range(nDiv):
        minAreaIdx_i = (poolArea[0, 1] - poolArea[0, 0]) * (div_i / np.float(nDiv)) + poolArea[0, 0]
        maxAreaIdx_i = (poolArea[0, 1] - poolArea[0, 0]) * ((div_i + 1) / np.float(nDiv)) + poolArea[0, 0]
        for div_j in range(nDiv):
            minAreaIdx_j = (poolArea[1, 1] - poolArea[1, 0]) * (div_j / np.float(nDiv)) + poolArea[1, 0]
            maxAreaIdx_j = (poolArea[1, 1] - poolArea[1, 0]) * ((div_j + 1) / np.float(nDiv)) + poolArea[1, 0]
            poolArea_i = np.array([[minAreaIdx_i, minAreaIdx_j],[maxAreaIdx_i, maxAreaIdx_j]]).transpose()
            # print(poolArea_i)
            vec = pool(FeatLocs, FeatVecs, poolArea_i, poolMethod)
            poolVec_tmp.append(vec * levelWs[0])
            poolVec.append(vec * levelWs[0])

    poolVec_tmp = np.array(poolVec_tmp)
    # print(poolVec_tmp.shape)
    for level_r in range(nLevel-1):
        level_i = level_r + 1
        nDiv = nDiv / 2
        poolVec_foo = []
        for div_i in range(nDiv):
            for div_j in range(nDiv):
                idxs_i = [div_i*(nDiv*2*2)+div_j*nDiv, div_i*(nDiv*2*2)+div_j*nDiv+1, div_i*(nDiv*2*2)+nDiv*2+div_j*nDiv, div_i*(nDiv*2*2)+nDiv*2+div_j*nDiv+1]
                # print(idxs_i)
                tmp = poolVec_tmp[idxs_i,:]
                if poolMethod == 'mean':
                    vec = np.mean(tmp, 0)
                elif poolMethod == 'max':
                    vec = np.max(tmp, 0)
                elif poolMethod == 'min':
                    vec = np.min(tmp, 0)
                elif poolMethod == 'sum':
                    vec = np.sum(tmp, 0)
                # print(vec.shape)
                vec = vec * levelWs[level_i] / levelWs[level_r]
                poolVec_foo.append(vec)
                poolVec.append(vec)
        poolVec_tmp = np.array(poolVec_foo)

    poolVec = np.array(poolVec)
    poolVec = poolVec.reshape([-1])
    return poolVec


# def main():
#     m = sio.loadmat('feat.mat')
#
#     FeatLocs = m['FeatLocs']
#     FeatVecs = m['FeatVecs']
#     poolArea = np.array([[0, 29], [0, 19]])
#     levelWs = np.array([1, 1, 1])
#
#     poolVecs = pyrPool(FeatLocs, FeatVecs, poolArea, 'mean', levelWs)
#     print(poolVecs.shape)
# if __name__ == '__main__':
#     main()