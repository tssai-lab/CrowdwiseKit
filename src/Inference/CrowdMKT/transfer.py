import warnings

import numpy as np
from sklearn import linear_model
import sys

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf, suppress=True)


def dict_update(y, d, x, n_components):
   
    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        
        d[:, i] = 0
       
        r = (y - np.dot(d, x))[:, index]
       
        u, s, v = np.linalg.svd(r, full_matrices=False)
       
        d[:, i] = u[:, 0]
       
        for j, k in enumerate(index):
            x[i, k] = s[0] * v[0, j]
    return d, x


def normalize(data):
   
    '''
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = -data[i][j]
    '''
    data = data.T
    mx = np.max(data)
    mn = np.min(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j] - mn) / (mx - mn)

    return data.T


def k_svd(matrix, bases, iterations, tolerances, gammas):
   
    train_data = matrix  

    u, s, v = np.linalg.svd(train_data)
    n_comp = bases  
    dict_data = u[:, :n_comp]

    max_iter = iterations 
    dictionary = dict_data

    y = train_data

    tolerance = tolerances
    before = 0
    now = 0
    gamma = gammas

    for i in range(max_iter):
       
        x = linear_model.orthogonal_mp(dictionary, y)
        e = np.linalg.norm(y - np.dot(dictionary, x))

        dict_update(y, dictionary, x, n_comp)

        
        sparse_code = linear_model.orthogonal_mp(dictionary, y)
        train_restruct = dictionary.dot(sparse_code)

        L1 = 0  
        for j in range(len(sparse_code)):
            L1 += np.linalg.norm(sparse_code[j], 1)

        print("epoch{}:,matrix = {},L1 = {}".format(i + 1, pow(np.linalg.norm(train_data - train_restruct, "fro"), 2, ), L1))

        now = pow(np.linalg.norm(train_data - train_restruct, "fro"), 2, ) + L1 * gamma
        if i > 0:  
            if abs((before - now) / before) < tolerance:
                print("Loss is stable, iteration stops")
                print("The shape：", sparse_code.shape)
                break
            if i == max_iter - 1:
                print("The iteration stops！")
        before = now

   
    sparse_code = linear_model.orthogonal_mp(dictionary, y)
    train_restruct = dictionary.dot(sparse_code)


    features=[]
    id=1
    for y in sparse_code:
        y=np.insert(y,0,id)
        y=y.tolist()
        features.append(y)
        id = id +1
    fp=open(transfername,'w')
    for j in range(len(features)):
        fp.write(' '.join('%s'% k for k in features[j])+'\n')

    return sparse_code


if __name__ == '__main__':
    source = sys.argv[1]
    target = sys.argv[2]

    transfername = sys.argv[3]
    target_vec = np.loadtxt(target)
    target_vec = target_vec[:,1:].T
    source_vec = np.loadtxt(source)
    source_vec = source_vec[:,1:].T

    data_multi_trans = np.hstack((target_vec,source_vec))

    data_multi_trans = normalize(data_multi_trans)

    k_svd(data_multi_trans, 20, 1000, 1e-6, 0.1)
