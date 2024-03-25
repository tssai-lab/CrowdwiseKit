# 主程序

import torch
import os
from utils.netVisualization import visualize_graph
from utils.netVisualization import visualize_embedding
import time
import numpy as np

from utils.datasetCreate import datasetCreate
from model.CGNNATModel import CGNNATModel
from model.CGNNATModel import CGNNAT_CLModel

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import BCELoss
from torch.optim import SGD
from torch import optim

device = torch.device('cuda')


'''
使用真实标签进行CGNNAT模型训练,不引入众包层
'''


def evalute(model, data):
    model.eval()
    correct_test = 0
    correct_train = 0
    total_test = len(data['true_label'][data['test_mask']])
    total_train = len(data['true_label'][data['train_mask']])
    with torch.no_grad():
        out, h = model(data['x'],data['adj'])
        pred = out.argmax(dim=1)
    correct_test += torch.eq(pred[data['test_mask']], data['true_label'][data['test_mask']].long()).sum().float().item()
    correct_train += torch.eq(pred[data['train_mask']], data['true_label'][data['train_mask']].long()).sum().float().item()
    return correct_test / total_test, correct_train / total_train

ratio_train = 0.7  # 70%训练数据

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'oak'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/vehicle/processed/'
# ds_name = 'vehicle'

root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/'
ds_name = 'waveform'

ds = datasetCreate(root, ds_name)
pt_path = root + ds_name + '.pt'
if not os.path.exists(pt_path):
    ds.process()
data = ds.get()

nsample = data['nsample']
print(nsample)
threshold = nsample * ratio_train
train_mask = []
test_mask = []
for i in range(nsample):
    if i < threshold:
        train_mask.append(i)
    else:
        test_mask.append(i)
data['train_mask'] = train_mask
data['test_mask'] = test_mask

config = dict()
config['input_dim'] = data['x'].shape[1]
config['output_dim'] = torch.max(data['true_label']).item() + 1
config['hidden_dim1'] = 32
config['hidden_dim2'] = 16
config['hidden_dim3'] = 10
config['dropout'] = 0.3
config['alpha'] = 0.05
config['concat'] = False
config['bias'] = True
config['nworker'] = data['nworker']
print('config: ', config)

cgat = CGNNATModel(config)
criterion = CrossEntropyLoss()
optimizer = Adam(cgat.parameters(), lr = 0.01)


def train(data):
    optimizer.zero_grad()
    out, h = cgat(data['x'], data['adj'])
    loss = criterion(out[data['train_mask']], data['true_label'][data['train_mask']].long())
    loss.backward()
    optimizer.step()
    return loss, out, h


loss_list_alder = []
for epoch in range(501):
    cgat.train()
    loss, out, h = train(data)
    loss_list_alder.append(loss.item())
    # print('\r loss = ', loss)
    acc_test, acc_train = evalute(cgat, data)
print('acc_test = ', acc_test)
print('acc_train = ', acc_train)


'''
使用众包标签进行CGNNAT模型训练,引入众包层
'''


def crowdevalute(model, data):
    model.eval()
    correct_test = 0
    correct_train = 0
    total_test = len(data['true_label'][data['test_mask']])
    total_train = len(data['true_label'][data['train_mask']])
    with torch.no_grad():
        out, h, output = model(data['x'],data['adj'])
        pred = output.argmax(dim=1)
    correct_test += torch.eq(pred[data['test_mask']], data['true_label'][data['test_mask']].long()).sum().float().item()
    correct_train += torch.eq(pred[data['train_mask']], data['true_label'][data['train_mask']].long()).sum().float().item()
    return correct_test / total_test, correct_train / total_train


# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'alder'
# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'eucalyptus'
# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'maple'
# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'oak'
# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/'
# ds_name = 'biodeg'
# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/'
# ds_name = 'ionosphere'
# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/vehicle/processed/'
# ds_name = 'vehicle'
root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/'
ds_name = 'waveform'


ds = datasetCreate(root, ds_name)
pt_path = root + ds_name + '.pt'
if not os.path.exists(pt_path):
    ds.process()
data = ds.get()


ratio_train = 0.7
nsample = data['nsample']
print(nsample)
threshold = nsample * ratio_train
train_mask = []
test_mask = []
for i in range(nsample):
    if i < threshold:
        train_mask.append(i)
    else:
        test_mask.append(i)
data['train_mask'] = train_mask
data['test_mask'] = test_mask


crowdresp = data['x_crowd']
crowd_train_mask = torch.where(crowdresp!=-1)
crowd_train_mask_dim1 = []
crowd_train_mask_dim2 = []
for i in range(len(crowd_train_mask[0])):
    crowd_train_mask_dim1.append(crowd_train_mask[0][i].item())
    crowd_train_mask_dim2.append(crowd_train_mask[1][i].item())
data['crowd_train_mask_dim1'] = crowd_train_mask_dim1
data['crowd_train_mask_dim2'] = crowd_train_mask_dim2


config = dict()
config['input_dim'] = data['x'].shape[1]
config['output_dim'] = torch.max(data['true_label']).item() + 1
config['hidden_dim1'] = 32
config['hidden_dim2'] = 16
config['hidden_dim3'] = 10
config['dropout'] = 0.3
config['alpha'] = 0.05
config['concat'] = False
config['bias'] = True
config['nworker'] = data['nworker']
config['nclass'] = config['output_dim']

cgcl = CGNNAT_CLModel(config)
# out, h, output = cgcl(data['x'], data['adj'])
criterion = CrossEntropyLoss()
optimizer = Adam(cgcl.parameters(), lr = 0.01)


def train(data):
    optimizer.zero_grad()
    out, h, output = cgcl(data['x'], data['adj'])
    loss = criterion(out[crowd_train_mask_dim1,crowd_train_mask_dim2], crowdresp[crowd_train_mask_dim1,crowd_train_mask_dim2].long())
    loss.backward()
    optimizer.step()
    return loss, out, h


loss_list = []
acc_list = []
for epoch in range(501):
    cgcl.train()
    loss, out, h = train(data)
    # print('\r loss = ', loss)
    acc_test, acc_train = crowdevalute(cgcl, data)
    if epoch % 10 == 0:
        # print('\r loss = ', loss)
        loss_list.append(loss.item())
        acc_list.append(acc_test)
print('acc_test = ', 1-acc_test)
print('acc_train = ', 1-acc_train)

loss_save_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/loss_waveform.pkl'
acc_save_path = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/results/loss/acc_waveform.pkl'
torch.save(loss_list, loss_save_path)
torch.save(acc_list, acc_save_path)


'''
使用集成标签进行CGNNAT模型训练,不引入众包层
'''


def evalute(model, data):
    model.eval()
    correct_test = 0
    correct_train = 0
    total_test = len(data['true_label'][data['test_mask']])
    total_train = len(data['true_label'][data['train_mask']])
    with torch.no_grad():
        out, h = model(data['x'],data['adj'])
        pred = out.argmax(dim=1)
    correct_test += torch.eq(pred[data['test_mask']], data['true_label'][data['test_mask']].long()).sum().float().item()
    correct_train += torch.eq(pred[data['train_mask']], data['true_label'][data['train_mask']].long()).sum().float().item()
    return correct_test / total_test, correct_train / total_train

ratio_train = 0.7  # 70%训练数据

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'alder'
# ds_name = 'eucalyptus'
# ds_name = 'maple'
# ds_name = 'oak'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/'
# ds_name = 'biodeg'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/'
# ds_name = 'ionosphere'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/vehicle/processed/'
# ds_name = 'vehicle'

root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/'
ds_name = 'waveform'

ds = datasetCreate(root, ds_name)
pt_path = root + ds_name + '.pt'
if not os.path.exists(pt_path):
    ds.process()
data = ds.get()

nsample = data['nsample']
print(nsample)
threshold = nsample * ratio_train
train_mask = []
test_mask = []
for i in range(nsample):
    if i < threshold:
        train_mask.append(i)
    else:
        test_mask.append(i)
data['train_mask'] = train_mask
data['test_mask'] = test_mask

infer_path = root + ds_name + '.mv.infer.txt'
# infer_path = root + ds_name + '.ds.infer.txt'
# infer_path = root + ds_name + '.glad.infer.txt'
# infer_path = root + ds_name + '.kos.infer.txt'
# infer_path = root + ds_name + '.ibcc.infer.txt'
# infer_path = root + ds_name + '.plat.infer.txt'

idx_features_labels = np.genfromtxt(infer_path, delimiter='')
y = idx_features_labels[:, -1]
y = torch.tensor(y, dtype=torch.int32)

data['infer_label'] = y


config = dict()
config['input_dim'] = data['x'].shape[1]
config['output_dim'] = torch.max(data['true_label']).item() + 1
config['hidden_dim1'] = 64
config['hidden_dim2'] = 32
config['hidden_dim3'] = 16
config['dropout'] = 0.3
config['alpha'] = 0.05
config['concat'] = False
config['bias'] = True
config['nworker'] = data['nworker']
print('config: ', config)

cgat = CGNNATModel(config)
criterion = CrossEntropyLoss()
optimizer = Adam(cgat.parameters(), lr = 0.01)

def train(data):
    optimizer.zero_grad()
    out, h = cgat(data['x'], data['adj'])
    loss = criterion(out[data['train_mask']], data['infer_label'][data['train_mask']].long())
    loss.backward()
    optimizer.step()
    return loss, out, h


loss_list_alder = []
for epoch in range(501):
    cgat.train()
    loss, out, h = train(data)
    loss_list_alder.append(loss.item())
    # print('\r loss = ', loss)
    acc_test, acc_train = evalute(cgat, data)
print('acc_test = ', acc_test)
print('acc_train = ', acc_train)


'''
使用集成标签进行SVM模型训练
'''


from sklearn import svm

ratio_train = 0.7  # 70%训练数据

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/real-world/leaves/processed/'
# ds_name = 'alder'
# ds_name = 'eucalyptus'
# ds_name = 'maple'
# ds_name = 'oak'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/biodeg/processed/'
# ds_name = 'biodeg'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/ionosphere/processed/'
# ds_name = 'ionosphere'

# root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/vehicle/processed/'
# ds_name = 'vehicle'

root = 'E:/wm/gitee_workspace/crowdsourcing-tools/CrowdGNN-Attention/Data/synthetic/waveform/processed/'
ds_name = 'waveform'

ds = datasetCreate(root, ds_name)
pt_path = root + ds_name + '.pt'

if not os.path.exists(pt_path):
    ds.process()
data = ds.get()

nsample = data['nsample']
print(nsample)
threshold = nsample * ratio_train
train_mask = []
test_mask = []
for i in range(nsample):
    if i < threshold:
        train_mask.append(i)
    else:
        test_mask.append(i)
data['train_mask'] = train_mask
data['test_mask'] = test_mask

infer_path = root + ds_name + '.mv.infer.txt'
# infer_path = root + ds_name + '.ds.infer.txt'
# infer_path = root + ds_name + '.glad.infer.txt'
# infer_path = root + ds_name + '.kos.infer.txt'
# infer_path = root + ds_name + '.ibcc.infer.txt'
# infer_path = root + ds_name + '.plat.infer.txt'

idx_features_labels = np.genfromtxt(infer_path, delimiter='')
y = idx_features_labels[:, -1]
y = torch.tensor(y, dtype=torch.int32)

data['infer_label'] = y

clf = svm.SVC()
clf.fit(data['x'][data['train_mask']], data['infer_label'][data['train_mask']])
print(clf.score(data['x'][data['test_mask']], data['true_label'][data['test_mask']]))