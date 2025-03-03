
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import MetaModule, MetaLinear, MetaConv2d, MetaBatchNorm2d, MetaParameter

from .resnet import ResNet18, ResNet34


class CL_model(MetaModule):
    def __init__(self, arch, nclass, in_dim, nb_annotator):
        super(CL_model, self).__init__()
        self.nclass = nclass
        self.nb_annotator = nb_annotator
        self.CM_CL = MetaParameter(self.__ident_init(), requires_grad=True)
        if arch == 'resnet34':
            self.backbone = ResNet34(num_classes=nclass, in_dim=in_dim)
        elif arch == 'resnet18':
            self.backbone = ResNet18(num_classes=nclass, in_dim=in_dim)
        else:
            pass

    def forward_CL(self, x):
        x = self.backbone(x)
        x = F.softmax(x, dim=-1)
        out = torch.log(F.softmax(self.CM_CL(x), dim=-1) + 1e-5)

        return out, x

    def forward_CCC(self, x, CM_CCC):
        x = self.backbone(x)
        x = F.softmax(x, dim=-1)
        out = torch.log(F.softmax(torch.einsum('ij,kjl->ikl', x, (self.CM_CL.weight + CM_CCC)), dim=-1) + 1e-5)

        return out, x

    def forward(self, x):
        '''get the logits'''
        x = self.backbone(x)
        return x

    def __ident_init(self):
        CM_CL = torch.cat([torch.eye(self.nclass).unsqueeze(0) for i in range(self.nb_annotator)], dim=0).cuda()
        return CM_CL
    def get_mw_params(self):
        CM_CL = self.CM_CL.weight.detach()
        return CM_CL