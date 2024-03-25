
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolutionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # input: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(self.dropout(inputs), self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2*out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain= 1.414)
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
            h: (N, in_features)
            adj: sparse matrix with shape (N,N)
            :return:
        '''
        Wh = torch.mm(h, self.W) # (N, out_feature)
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :]) # (N, 1)
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :]) # (N, 1)

        # Wh1 + Wh2.T 是N*N矩阵，第i行第j列是Wh1[i]+Wh2[j]
        # 那么Wh1 + Wh2.T的第i行第j列刚好就是文中的a^T*[Whi||Whj]
        # 代表着节点i对节点j的attention
        e = self.leakyrelu(Wh1 + Wh2.T)
        padding = (-2 ** 31) * torch.ones_like(e)
        attention = torch.where(adj > 0, e, padding)
        attention = F.softmax(attention, dim = 1)

        attention = F.dropout(attention, self.dropout, training = self.training)

        return attention


class CGNNATModel(torch.nn.Module):
    def __init__(self, config):
        super(CGNNATModel, self).__init__()
        torch.manual_seed(123)

        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim1 = config['hidden_dim1']
        self.hidden_dim2 = config['hidden_dim2']
        self.hidden_dim3 = config['hidden_dim3']
        self.dropout = config['dropout']
        self.alpha = config['alpha']
        self.concat = config['concat']
        self.bias = config['bias']
        self.nworker = config['nworker']

        self.gc1 = GraphConvolutionLayer(self.input_dim, self.hidden_dim1, self.dropout, self.bias)
        self.gc2 = GraphConvolutionLayer(self.hidden_dim1, self.hidden_dim2, self.dropout, self.bias)
        self.gc3 = GraphConvolutionLayer(self.hidden_dim2, self.hidden_dim2, self.dropout, self.bias)
        self.gc4 = GraphConvolutionLayer(self.hidden_dim2, self.hidden_dim3, self.dropout, self.bias)
        # self.gc5 = GraphConvolutionLayer(self.hidden_dim3, self.hidden_dim3, self.dropout, self.bias)
        self.ga1 = GraphAttentionLayer(self.hidden_dim2, self.hidden_dim2, self.dropout, self.alpha, self.concat)
        # self.ga2 = GraphAttentionLayer(self.hidden_dim2, self.hidden_dim2, self.dropout, self.alpha, self.concat)
        # self.ga3 = GraphAttentionLayer(self.hidden_dim3, self.hidden_dim3, self.dropout, self.alpha, self.concat)
        self.classifier = Linear(self.hidden_dim3, self.output_dim)

    def  forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        attn_adj = self.ga1(x, adj)
        x = F.relu(self.gc3(x, attn_adj))
        attn_adj = self.ga1(x, adj)
        h = F.relu(self.gc4(x, attn_adj))
        t = self.classifier(h)
        output = F.softmax(t, dim = 1)

        return output, h


class CGNNAT_CLModel(torch.nn.Module):
    def __init__(self, config):
        super(CGNNAT_CLModel, self).__init__()
        torch.manual_seed(123)

        self.config = config
        self.base_model = CGNNATModel(self.config)
        self.cl = CrowdsourcingLayer(self.config)

    def  forward(self, x, adj):
        output, h = self.base_model(x, adj)
        h = self.cl(output)
        out = F.softmax(h, dim=2)
        return out, h, output


class CrowdsourcingLayer(torch.nn.Module):
    def __init__(self, config):
        super(CrowdsourcingLayer, self).__init__()
        self.nclass = config['nclass']
        self.nworker = config['nworker']

        self.W = nn.Parameter(torch.Tensor(self.nworker, self.nclass, self.nclass))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        Wh = torch.matmul(x, self.W)
        return Wh


# class MaskedMultiCrossEntropy(nn.Module):
#     def __init__(self):
#         super(MaskedMultiCrossEntropy, self).__init__()
#         self.CELoss = CrossEntropyLoss(weight = torch.FloatTensor([1, 2]))
#
#     def forward(self, pred, target):
#         loss = self.CELoss(pred, target)


# class GCN(nn.Module):
#     def __init__(self, n_features, hidden_dim, dropout, n_classes):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolutionLayer(n_features, hidden_dim, dropout)
#         self.gc2 = GraphConvolutionLayer(hidden_dim, n_classes, dropout)
#         self.relu = nn.ReLU()
#
#     def forward(self, inputs, adj):
#         x = inputs
#         x = self.relu(self.gc1(x, adj))
#         x = self.gc2(x, adj)
#         return x