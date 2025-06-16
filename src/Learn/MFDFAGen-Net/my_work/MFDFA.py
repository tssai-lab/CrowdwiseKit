from vgg import *
import numpy as np
import torch.nn.functional as F
from functional import *


class MFDFA(nn.Module):
    def __init__(self,
                 num_annotators,
                 input_dims,
                 num_class,
                 rate=0.5,
                 conn_type='MW',
                 backbone_model=None,
                 user_feature=None,
                 common_module='simple',
                 input=None,
                 num_side_features=None,
                 nb=None,
                 u_features=None,
                 v_features=None,
                 u_features_side=None,
                 v_features_side=None,
                 input_dim=None,
                 emb_dim=None,
                 hidden=None,
                 gumbel_common=False):

        super(MFDFA, self).__init__()
        self.annotator_count = num_annotators
        self.connection_scheme = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)

        # 严格保持原始特征提取路径
        self.linear1 = nn.Linear(input_dims, 128)
        self.linear2 = nn.Linear(128, num_class)  # 直接连接到输出层

        # 保持原始dropout配置
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # 配置参数
        self.mixing_ratio = rate

        # 初始化转换矩阵（保持原始实现）
        self.annotator_transform = nn.Parameter(
            self._init_identity_matrix((num_annotators, num_class, num_class)),
            requires_grad=True
        )
        self.instance_transform = nn.Parameter(
            self._init_identity_matrix((num_class, num_class)),
            requires_grad=True
        )

        # 主干网络初始化
        self.backbone = None
        if backbone_model == 'vgg16':
            self.backbone = VGG('VGG16').cuda()

        # 双特征处理模块（保持原始结构）
        self.dual_module_type = common_module
        if self.dual_module_type == 'simple':
            self.dual_feature_size = 20
            self.annotator_features = torch.from_numpy(user_feature).float().cuda()

            # 实例难度路径（保持原始输入维度）
            self.difficulty_net = nn.Sequential(
                nn.Linear(num_class, 128),  # 关键修复：使用num_class而非固定值8
                nn.ReLU(),
                nn.Linear(128, self.dual_feature_size)
            )

            # 标注者特征路径
            self.annotator_net = nn.Linear(self.annotator_features.size(1), self.dual_feature_size)

            # 注意力机制（保持原始配置）
            self.attention_heads = 4
            self.head_dim = self.dual_feature_size // self.attention_heads
            assert self.head_dim * self.attention_heads == self.dual_feature_size, \
                "Feature size must be divisible by number of attention heads"

            self.query_proj = nn.Linear(self.dual_feature_size, self.dual_feature_size)
            self.key_proj = nn.Linear(self.dual_feature_size, self.dual_feature_size)

    def _init_identity_matrix(self, shape):
        """初始化单位式转换矩阵（保持原始实现）"""
        identity = np.zeros(shape)
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    identity[r, i, i] = 2.0
        elif len(shape) == 2:
            for i in range(shape[1]):
                identity[i, i] = 2.0
        return torch.Tensor(identity).cuda()

    def _process_dual_features(self, instance_data):
        """处理实例和标注者特征（保持原始计算流程）"""
        # 处理实例难度特征
        instance_vec = self.difficulty_net(instance_data)
        instance_vec = F.normalize(instance_vec, dim=1)

        # 处理标注者特征
        annotator_vec = self.annotator_net(self.annotator_features)
        annotator_vec = F.normalize(annotator_vec, dim=1)

        batch_size = instance_vec.size(0)

        # 多头注意力转换（保持原始einsum计算）
        Q = self.query_proj(instance_vec).view(batch_size, self.attention_heads, self.head_dim)
        K = self.key_proj(annotator_vec).view(self.annotator_count, self.attention_heads, self.head_dim)
        K = K.permute(1, 0, 2)  # [heads, annotators, dim]

        # 注意力评分（保持原始计算方式）
        attention_logits = torch.einsum('bhd,had->bha', Q, K) / (self.head_dim ** 0.5)
        attention_weights = attention_logits.mean(dim=1)

        return torch.sigmoid(attention_weights)

    def forward(self, input_data, y=None, mode='train', support=None, support_t=None, idx=None):
        # 特征提取（严格保持原始路径）
        if self.backbone:
            base_output = self.backbone(input_data)
        else:
            flattened = input_data.view(input_data.size(0), -1)
            x = self.relu(self.linear1(flattened))
            x = self.dropout1(x)  # 保持原始dropout位置
            x = self.linear2(x)  # 直接连接到输出层
            base_output = x  # 不在中间应用softmax

        # 最终分类输出（保持原始位置应用softmax）
        cls_out = F.softmax(base_output, dim=1)
        crowd_output = None

        if mode == 'train':
            # 使用softmax前的特征作为双特征模块输入（关键修复）
            # 保持原始视图操作：instance_x = cls_out.view(cls_out.size(0), -1)
            adjustment_weights = self._process_dual_features(base_output)

            # 保持原始einsum计算
            instance_probs = torch.einsum('ij,jk->ik', (cls_out, self.instance_transform))
            annotator_probs = torch.einsum('ik,jkl->ijl', (cls_out, self.annotator_transform))

            # 组合预测（保持原始计算方式）
            crowd_output = (
                    adjustment_weights[:, :, None] * instance_probs[:, None, :] +
                    (1 - adjustment_weights[:, :, None]) * annotator_probs
            )
            crowd_output = crowd_output.transpose(1, 2)

        return cls_out, crowd_output