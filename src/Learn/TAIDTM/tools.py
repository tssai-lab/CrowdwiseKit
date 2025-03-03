import os
import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed, num_worker): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (0.6 - n) / norm_std, loc=n, scale=norm_std) #1-n
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    #print(P)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label=np.array([np.random.choice(l, size=num_worker, replace=True, p=P[i]) for i in range(labels.shape[0])])
    #record = [[0 for _ in range(label_num)] for i in range(label_num)]

    # for a, b in zip(labels, new_label):
        # a, b = int(a), int(b)
        # record[a][b] += 1


    # pidx = np.random.choice(range(P.shape[0]), 1000)
    # cnt = 0
    # for i in range(1000):
        # if labels[pidx[i]] == 0:
            # a = P[pidx[i], :]
            # cnt += 1
        # if cnt >= 10:
            # break
    return new_label#np.array(new_label)

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, -2).unsqueeze(-2)
    T_norm = row_abs / row_sum
    return T_norm



def fit(X, num_classes, percentage, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c)) # +1 -> index 
    eta_corr = X
    ind = []
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], percentage,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
            ind.append(idx_best)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
            
    return T, ind

def data_split(data, targets, true_targets,split_percentage, seed=1):
   
    num_samples = int(targets.shape[0])
    np.random.seed(int(seed))
    train_set_index = np.random.choice(num_samples, int(num_samples*split_percentage), replace=False)
    index = np.arange(data.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = data[train_set_index, :], data[val_set_index, :]
    train_labels, val_labels = targets[train_set_index], targets[val_set_index]
    train_true_labels, val_true_labels = true_targets[train_set_index], true_targets[val_set_index]

    return train_set, val_set, train_labels, val_labels, train_true_labels, val_true_labels


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-1)
            
    return net
    
    
def data_normal_2d(orign_data,dim=1):
    d_min = torch.min(orign_data,dim=dim)[0]
    if(dim==0):
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[:,idx] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]    
    else:
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    norm_data = (norm_data-0.5).true_divide(0.5)
    norm_data[norm_data.isnan()]=0
    return norm_data