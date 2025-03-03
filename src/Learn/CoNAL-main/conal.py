import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import math
from vgg import *
from torchvision import transforms
import torch.nn.functional as F
from functional import *


class CoNAL(nn.Module):
    def __identity_init(self, shape):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out).cuda()

    def __init__(self, num_annotators, input_dims, num_class, rate=0.5, conn_type='MW', backbone_model=None, user_feature=None
                 , common_module='simple', num_side_features=None, nb=None, u_features=None,
                 v_features=None, u_features_side=None, v_features_side=None, input_dim=None, emb_dim=None, hidden=None, gumbel_common=False):
        super(CoNAL, self).__init__()
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)

        self.linear1 = nn.Linear(input_dims, 128)

        self.ln1 = nn.Linear(128, 256)
        self.ln2 = nn.Linear(256, 128)

        self.linear2 = nn.Linear(128, num_class)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.rate = rate
        self.kernel = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class)),
                                   requires_grad=True)

        self.common_kernel = nn.Parameter(self.__identity_init((num_class, num_class)) ,
                                          requires_grad=True)

        self.backbone_model = None
        if backbone_model == 'vgg16':
            self.backbone_model = VGG('VGG16').cuda()
            self.feature = self.backbone_model.features
            self.classifier = self.backbone_model.classifier
        self.common_module = common_module

        if self.common_module == 'simple':
            com_emb_size = 20
            self.user_feature_vec = torch.from_numpy(user_feature).float().cuda()
            self.diff_linear_1 = nn.Linear(input_dims, 128)
            self.diff_linear_2 = nn.Linear(128, com_emb_size)
            self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)
            self.bn_instance = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.bn_user = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.single_weight = nn.Linear(20, 1, bias=False)

    def simple_common_module(self, input):
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        instance_difficulty = F.normalize(instance_difficulty)
        user_feature = self.user_feature_1(self.user_feature_vec)
        user_feature = F.normalize(user_feature)
        common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = torch.nn.functional.sigmoid(common_rate)
        return common_rate

    def forward(self, input, y=None, mode='train', support=None, support_t=None, idx=None):
        crowd_out = None
        if self.backbone_model:
            cls_out = self.backbone_model(input)
        else:
            x = input.view(input.size(0), -1)
            x = self.dropout1(F.relu(self.linear1(x)))
            x = self.linear2(x)
            cls_out = torch.nn.functional.softmax(x, dim=1)
        if mode == 'train':
            x = input.view(input.size(0), -1)
            if self.common_module == 'simple':
                common_rate = self.simple_common_module(x)
            common_prob = torch.einsum('ij,jk->ik', (cls_out, self.common_kernel))
            indivi_prob = torch.einsum('ik,jkl->ijl', (cls_out, self.kernel))

            crowd_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob   # single instance
            crowd_out = crowd_out.transpose(1, 2)
        if self.common_module == 'simple' or mode == 'test':
            return cls_out, crowd_out

class CoNAL_music(nn.Module):
    def __identity_init(self, shape):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out).cuda()

    def __init__(self, num_annotators, input_dims, num_class, rate=0.5, conn_type='MW', backbone_model=None, user_feature=None
                 , common_module='simple', num_side_features=None, nb=None, u_features=None,
                 v_features=None, u_features_side=None, v_features_side=None, input_dim=None, emb_dim=None, hidden=None, gumbel_common=False):
        super(CoNAL_music, self).__init__()
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)

        self.linear1 = nn.Linear(input_dims, 128)

        self.ln1 = nn.Linear(128, 256)
        self.ln2 = nn.Linear(256, 128)

        self.bn = torch.nn.BatchNorm1d(input_dims, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(128, affine=False)

        self.linear2 = nn.Linear(128, num_class)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.rate = rate
        self.kernel = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class)),
                                   requires_grad=True)

        self.common_kernel = nn.Parameter(self.__identity_init((num_class, num_class)) ,
                                          requires_grad=True)

        self.backbone_model = None
        if backbone_model == 'vgg16':
            self.backbone_model = VGG('VGG16').cuda()
            self.feature = self.backbone_model.features
            self.classifier = self.backbone_model.classifier
        self.common_module = common_module

        if self.common_module == 'simple':
            com_emb_size = 80
            self.user_feature_vec = torch.from_numpy(user_feature).float().cuda()
            self.diff_linear_1 = nn.Linear(input_dims, 128)
            self.diff_linear_2 = nn.Linear(128, com_emb_size)
            self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)
            self.bn_instance = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.bn_user = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.single_weight = nn.Linear(20, 1, bias=False)

    def simple_common_module(self, input):
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        user_feature = self.user_feature_1(self.user_feature_vec)
        user_feature = F.normalize(user_feature)
        common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = torch.nn.functional.sigmoid(common_rate)
        return common_rate

    def forward(self, input, y=None, mode='train', support=None, support_t=None, idx=None):
        crowd_out = None
        if self.backbone_model:
            cls_out = self.backbone_model(input)
        else:
            x = input.view(input.size(0), -1)
            x = self.bn(x)
            x = self.dropout1(F.relu(self.linear1(x)))
            x = self.bn1(x)
            x = self.linear2(x)
            cls_out = torch.nn.functional.softmax(x, dim=1)
        if mode == 'train':
            x = input.view(input.size(0), -1)
            if self.common_module == 'simple':
                common_rate = self.simple_common_module(x)
            elif self.common_module == 'gcn':
                u = list(range(self.num_annotators))
                common_rate, rec_out = self.gae(u, idx, support, support_t)
                common_rate = common_rate.transpose(0, 1)
            common_prob = torch.einsum('ij,jk->ik', (cls_out, self.common_kernel))
            indivi_prob = torch.einsum('ik,jkl->ijl', (cls_out, self.kernel))

            crowd_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob   # single instance
            crowd_out = crowd_out.transpose(1, 2)
        if self.common_module == 'simple' or mode == 'test':
            return cls_out, crowd_out




