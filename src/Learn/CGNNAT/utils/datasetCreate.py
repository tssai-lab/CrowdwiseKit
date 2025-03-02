# 将数据集构建为程序支持的数据集
import os

import numpy as np
import torch
import torch.nn.functional as F

class datasetCreate:
    def __init__(self, root, name):
        '''
        :param root: 数据集根目录
        :param name: 数据集名称

        注意！注意！注意！ 35行与36行针对真实数据集和模拟数据集的区分
        '''
        super(datasetCreate, self).__init__()

        self.root = root
        self.name = name
        self.typelist = ['.fea','.resp', '.gold']
        self.raw_paths = []
        self.processed_dir = root
        for i in range(len(self.typelist)):
            self.raw_paths.append(self.root + self.name + self.typelist[i])

        nworker = 0
        nsample = 0
        sample_ID_map = dict()

        f = open(self.raw_paths[1])
        for line in f:
            strs = line.split()
            if int(strs[0]) > nworker:
                nworker = int(strs[0])
        # self.nworker = nworker + 1  # 真实数据集从0开始编号，所以需要+1，模拟数据集不需要
        self.nworker = nworker

        g = open(self.raw_paths[2])
        instance_id = 0
        for line in g:
            strs = line.split()
            sample_ID_map[int(strs[0])] = instance_id
            instance_id += 1
            nsample += 1
        if len(sample_ID_map) == nsample:
            self.nsample = nsample

        self.sample_ID_map = sample_ID_map

    def process(self):
        idx_features_labels = np.genfromtxt(self.raw_paths[0], delimiter=',')
        x = idx_features_labels[:, 1:-1]
        x = torch.tensor(x, dtype=torch.float32)
        y = idx_features_labels[:, -1]
        y = torch.tensor(y, dtype=torch.int32)

        crowd_responses = -torch.ones([self.nworker, self.nsample], dtype=torch.int32)
        f = open(self.raw_paths[1])
        for line in f:
            strs = line.split()
            crowd_responses[int(strs[0])-1][self.sample_ID_map[int(strs[1])]] = int(strs[2])

        adj = torch.zeros([self.nsample, self.nsample], dtype=torch.float32)
        for i in range(self.nsample):
            for j in range(self.nsample):
                adj[i][j] = F.cosine_similarity(x[i], x[j], dim=0)

        data = dict()
        data['x'] = x
        data['true_label'] = y
        data['sample_ID_map'] = self.sample_ID_map
        data['nworker'] = self.nworker
        data['nsample'] = self.nsample
        data['x_crowd'] = crowd_responses
        data['adj'] = adj
        torch.save(data, os.path.join(self.processed_dir, self.name + '.pt'))

    def len(self):
        idx_features_labels = np.genfromtxt(self.raw_paths[0], delimiter=',', dtype=np.int32)
        uid = idx_features_labels[:, 0]
        return len(uid)

    def get(self):
        data = torch.load(os.path.join(self.processed_dir, self.name + '.pt'))
        return data


if __name__ == '__main__':
    # root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
    # ds_name = 'oak'
    # root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/'
    # ds_name = 'biodeg'
    root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/'
    ds_name = 'ionosphere'
    ds = datasetCreate(root, ds_name)
    ds.process()
    data = ds.get()
    print(data)