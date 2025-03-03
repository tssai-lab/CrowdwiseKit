import torch
import torch.nn as nn
import torch.nn.functional as F
from common_DNNTM import Config
import numpy as np
import torch
import copy
from resnet import *
import math
#torch.cuda.set_device(Config.device_id)

class left_neural_net(nn.Module):
    def __init__(self,nclass=Config.num_classes):
        super(left_neural_net, self).__init__()
        self.backbone = ResNet18_F(nclass)

    def forward(self, x):
        h,_ = self.backbone(x)

        return torch.nn.functional.softmax(h,dim=-1)
            
class left_neural_aux_net_one_source(nn.Module):
    def __init__(self,num_class=Config.num_classes, fea_dim=512,linear_dim=64):
        super(left_neural_aux_net_one_source, self).__init__()
        
        self.num_class = num_class
        self.backbone_for_NT = ResNet18_F(num_class)
        self.linear_1 = nn.Linear(fea_dim,linear_dim)
        self.fea_to_NT_layer = fea_to_NT(num_class=num_class, linear_dim=linear_dim)
        #self.bias=nn.Parameter(torch.zeros((num_class,num_class),requires_grad=True))

    def forward(self, x):
        _,x=self.backbone_for_NT(x)
        x = F.relu(self.linear_1(x))
        
        noise_matrices_for_x = self.fea_to_NT_layer(x)
        return noise_matrices_for_x

    def copy_backbone(self, param_dict,param_dict2):
        self.backbone_for_NT.load_state_dict(param_dict)
        self.linear_1.load_state_dict(param_dict2)
        return
        
    def copy_NT_layer(self, param_dict):
        self.fea_to_NT_layer.load_state_dict(param_dict)
        return

    # def copy_bias(self, param_dict):
        # self.bias.data =  param_dict
        # self.bias.requires_grad = True
        # return

class left_neural_aux_net_multi_source_with_fea_to_NT_layers(nn.Module):
    def __init__(self,num_class=Config.num_classes, fea_dim=512, worker_num=Config.expert_num, linear_dim=64):
        super(left_neural_aux_net_multi_source_with_fea_to_NT_layers, self).__init__()
        
        self.num_class = num_class
        self.worker_num = worker_num
        self.backbone_for_NT = ResNet18_F(num_class)
        self.linear_1 = nn.Linear(fea_dim,linear_dim)
        for r in range(self.worker_num):
           m_name = "worker"+str(r)
           self.add_module(m_name,fea_to_NT(num_class=num_class, linear_dim=linear_dim))
        #self.bias=nn.Parameter(torch.zeros((num_class,num_class),requires_grad=True))

    def forward(self, x, worker_id=-1, no_grad=False):
        if(no_grad):
            with torch.no_grad():
                _,x=self.backbone_for_NT(x)
                x=F.relu(self.linear_1(x))
        else:
            _,x=self.backbone_for_NT(x)
            x=F.relu(self.linear_1(x))

        if(worker_id<0):
            noise_matrices_for_x = torch.zeros((x.size(0),self.worker_num,self.num_class,self.num_class)).cuda()
            for r in range(self.worker_num):
               m_name = "worker"+str(r)
               module = getattr(self,m_name)
               noise_matrices_for_x[:,r,:,:]=  module(x)#,self.bias)  
            return noise_matrices_for_x
        else:
            m_name = "worker"+str(worker_id)
            module = getattr(self,m_name)
            noise_matrices_for_x=  module(x)#,self.bias)  
            return noise_matrices_for_x        
    
    def copy_backbone(self, param_dict,param_dict2):
        self.backbone_for_NT.load_state_dict(param_dict)
        self.linear_1.load_state_dict(param_dict2)
        return
        
    def copy_NT_layer(self, param_dict):
        for r in range(self.worker_num):
           m_name = "worker"+str(r)
           module = getattr(self,m_name)
           module.load_state_dict(param_dict)
        return
        
    def get_NT_layer(self, worker_id):
        m_name = "worker"+str(worker_id)
        module = getattr(self,m_name)    
        return module

    # def copy_bias(self, param_dict):
        # self.bias.data =  param_dict
        # self.bias.requires_grad = True
        # return

class fea_to_NT(nn.Module):
    def __init__(self,num_class=Config.num_classes, fea_dim=512, linear_dim=64):
        super(fea_to_NT, self).__init__()

        self.num_class=num_class
        self.to_NT=nn.Linear(linear_dim, num_class*num_class, bias=True)

    def forward(self, x, no_softmax=False):
        
        noise_matrices_for_x=  self.to_NT(x).view(-1,self.num_class,self.num_class)

        if(no_softmax):
            return noise_matrices_for_x
        noise_matrices_for_x= F.softmax(noise_matrices_for_x.cuda(), dim=-1)
        return noise_matrices_for_x



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class left_neural_aux_net_multi_source_with_gcn(nn.Module):
    def __init__(self, emb, adj, num_class=Config.num_classes,fea_dim=512, worker_num=Config.expert_num, linear_dim=64):
        super(left_neural_aux_net_multi_source_with_gcn, self).__init__()
        
        self.num_class = num_class
        self.worker_num = worker_num
        self.backbone_for_NT = ResNet18_F(num_class)
        self.linear_1 = nn.Linear(fea_dim,linear_dim)
        self.fea_to_NT_layer = fea_to_NT(num_class=num_class, linear_dim=linear_dim)
        #self.bias=nn.Parameter(torch.zeros((num_class,num_class),requires_grad=True))
        self.linear_dim = linear_dim
        self.gc1 = GraphConvolution(worker_num, 64)
        self.relu = nn.LeakyReLU(0.2)
        self.gc2 = GraphConvolution(64, linear_dim*num_class*num_class+num_class*num_class)
        #self.linear_2 = nn.Linear(128,linear_dim*num_class*num_class)
        #self.gc3 = GraphConvolution(128, linear_dim*num_class*num_class)
        
        self.emb=emb.float()
        self.adj=self.gen_adj(adj.float())

    def forward(self, x, no_grad=False):
        if(no_grad):
            with torch.no_grad():
                _,x=self.backbone_for_NT(x)
                x=F.relu(self.linear_1(x))
        else:
            _,x=self.backbone_for_NT(x)
            x=F.relu(self.linear_1(x))

        base_noise_matrices_for_x = self.fea_to_NT_layer(x,no_softmax=True).unsqueeze(1)  

        f = self.gc1(self.emb, self.adj)
        f = self.relu(f)
        f = self.gc2(f, self.adj)
        
        f = f.reshape(self.worker_num, self.linear_dim+1, self.num_class,self.num_class)
        f = torch.einsum("rijk,ni->nrjk",(f[:,:-1],x))+f[:,-1].unsqueeze(0)
        f = f #+ base_noise_matrices_for_x
        noise_matrices_for_x = F.softmax(f, dim=-1)
        return noise_matrices_for_x      
    
    def copy_backbone(self, param_dict,param_dict2):
        self.backbone_for_NT.load_state_dict(param_dict)
        self.linear_1.load_state_dict(param_dict2)
        return

    # def copy_bias(self, param_dict):
        # self.bias.data =  param_dict
        # self.bias.requires_grad = True
        # return

    def copy_NT_layer(self, param_dict):
        self.fea_to_NT_layer.load_state_dict(param_dict)
        return

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj
       