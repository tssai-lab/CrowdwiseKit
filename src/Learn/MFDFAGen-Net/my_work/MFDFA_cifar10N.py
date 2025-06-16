import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg16
from model.resnet import ResNet34
from model.resnet110 import ResNet110, BasicBlock
from vgg import VGG
from functional import GumbelSigmoid


class MFDFA_cifar10N(nn.Module):
    def __init__(self,
                 num_annotators,
                 input_dims,
                 num_classes,
                 rate=0.5,
                 connection_type='MW',
                 backbone_model=None,
                 user_features=None,
                 common_module_type='simple',
                 num_side_features=None,
                 nb=None,
                 u_features=None,
                 v_features=None,
                 u_side_features=None,
                 v_side_features=None,
                 input_dim=None,
                 embedding_dim=None,
                 hidden_dim=None,
                 use_gumbel=False):

        super(MFDFA_cifar10N, self).__init__()
        self.annotator_count = num_annotators
        self.connection_type = connection_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01) if use_gumbel else None

        # 严格保持原始特征提取路径 --------------------------------
        self.linear1 = nn.Linear(input_dims, 128)  # 关键修复：使用input_dims而非num_classes
        self.linear2 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # 保留批归一化层（虽然原始部分被注释）
        self.bn = nn.BatchNorm1d(input_dims, affine=False)
        self.bn1 = nn.BatchNorm1d(128, affine=False)
        # ------------------------------------------------------

        # 配置参数
        self.mixing_ratio = rate

        # 初始化转换矩阵
        self.annotator_transform = nn.Parameter(
            self._init_identity_tensor((num_annotators, num_classes, num_classes)),
            requires_grad=True
        )
        self.instance_transform = nn.Parameter(
            self._init_identity_tensor((num_classes, num_classes)),
            requires_grad=True
        )

        # 主干网络初始化
        self.backbone = None
        if backbone_model == 'resnet110':
            self.backbone = ResNet110(BasicBlock, [18, 18, 18]).cuda()

        # 双特征处理模块
        self.dual_module = common_module_type
        if self.dual_module == 'simple':
            self.dual_embed_size = 40
            self.annotator_features = torch.from_numpy(user_features).float().cuda()

            # 实例特征处理路径（保持原始结构）
            self.diff_linear_1 = nn.Linear(num_classes, 128)  # 关键修复：使用input_dims
            self.diff_linear_2 = nn.Linear(128, self.dual_embed_size)

            # 标注者特征处理路径
            self.user_feature_1 = nn.Linear(
                self.annotator_features.size(1),
                self.dual_embed_size
            )

            # 多头注意力机制
            self.attention_heads = 4
            self.head_dim = self.dual_embed_size // self.attention_heads
            assert self.head_dim * self.attention_heads == self.dual_embed_size, \
                "Embedding size must be divisible by number of attention heads"

            self.q_linear = nn.Linear(self.dual_embed_size, self.dual_embed_size)
            self.k_linear = nn.Linear(self.dual_embed_size, self.dual_embed_size)

    def _init_identity_tensor(self, shape):
        """创建类单位矩阵的张量"""
        identity = np.zeros(shape)
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    identity[r, i, i] = 2.0
        elif len(shape) == 2:
            for i in range(shape[1]):
                identity[i, i] = 2.0
        return torch.Tensor(identity).cuda()

    def _compute_dual_weights(self, instance_features):
        """计算特征权重（保持原始计算流程）"""
        # 实例特征提取
        instance_difficulty = self.diff_linear_1(instance_features)
        instance_difficulty = self.diff_linear_2(instance_difficulty)
        instance_difficulty = F.normalize(instance_difficulty, dim=1)

        # 用户特征提取
        user_feature = self.user_feature_1(self.annotator_features)
        user_feature = F.normalize(user_feature, dim=1)

        batch_size = instance_difficulty.size(0)

        # 多头注意力变换
        Q = self.q_linear(instance_difficulty).view(batch_size, self.attention_heads, self.head_dim)
        K = self.k_linear(user_feature).view(self.annotator_count, self.attention_heads, self.head_dim)
        K = K.permute(1, 0, 2)  # [heads, annotators, dim]

        # 注意力计算
        attention_scores = torch.einsum('bhd,had->bha', Q, K) / (self.head_dim ** 0.5)
        combined_scores = attention_scores.mean(dim=1)  # 多头平均

        return torch.sigmoid(combined_scores)

    def forward(self, input_data, targets=None, mode='train', support=None, support_t=None, idx=None):
        #
        if self.backbone:
            base_output = self.backbone(input_data)
        else:
            x = input_data.view(input_data.size(0), -1)

            # 保持原始处理流程（包括注释掉的BN层）
            # x = self.bn(x)  # 原始代码中注释掉了
            x = self.dropout1(self.relu(self.linear1(x)))
            # x = self.bn1(x)  # 原始代码中注释掉了
            logits = self.linear2(x)
            base_output = logits  # 不在中间应用softmax

        # 最终分类输出（保持原始位置应用softmax）
        cls_out = F.softmax(base_output, dim=1)
        crowd_output = None

        if mode == 'train':
            # 使用softmax前的特征作为双特征模块输入
            instance_features = base_output  # 原始使用 cls_out.view(...)

            if self.dual_module == 'simple':
                rectify_weights = self._compute_dual_weights(instance_features)

            #
            instance_probs = torch.einsum('ij,jk->ik', (cls_out, self.instance_transform))
            annotator_probs = torch.einsum('ik,jkl->ijl', (cls_out, self.annotator_transform))

            #
            crowd_output = (
                    rectify_weights[:, :, None] * instance_probs[:, None, :] +
                    (1 - rectify_weights[:, :, None]) * annotator_probs
            )
            crowd_output = crowd_output.transpose(1, 2)

        # 返回结果
        return cls_out, crowd_output