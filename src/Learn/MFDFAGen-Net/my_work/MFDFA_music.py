

from vgg import *
import numpy as np
from functional import *
# 从functional模块导入所有内容，可能是自定义的功能函数集合。

class MFDFA_music(nn.Module):
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

    def __init__(self, num_annotators, input_dims, num_class, rate=0.5, conn_type='MW', backbone_model=None,
                 user_feature=None, dual_module='simple', num_side_features=None, nb=None, u_features=None,
                 v_features=None, u_features_side=None, v_features_side=None, input_dim=None, emb_dim=None, hidden=None,
                 gumbel_common=False):
        super(MFDFA_music, self).__init__()
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)

        # 定义线性层
        self.linear1 = nn.Linear(input_dims, 128)
        self.ln1 = nn.Linear(128, 256)
        self.ln2 = nn.Linear(256, 128)
        self.bn = torch.nn.BatchNorm1d(input_dims, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(128, affine=False)
        self.linear2 = nn.Linear(128, num_class)

        # 定义dropout和激活函数
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # 率参数
        self.rate = rate

        # 众包建模的核参数
        self.kernel = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class)), requires_grad=True)
        self.instance_kernel = nn.Parameter(self.__identity_init((num_class, num_class)), requires_grad=True)

        # 主干模型设置
        self.backbone_model = None
        if backbone_model == 'vgg16':
            self.backbone_model = VGG('VGG16').cuda()
            self.feature = self.backbone_model.features
            self.classifier = self.backbone_model.classifier

        # 共同模块设置
        self.dual_module = dual_module
        if self.dual_module == 'simple':
            dual_emb_size = 80
            self.user_feature_vec = torch.from_numpy(user_feature).float().cuda()

            # 实例特征处理
            self.diff_linear_1 = nn.Linear(num_class, 128)
            self.diff_linear_2 = nn.Linear(128, dual_emb_size)

            # 用户特征处理
            self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), dual_emb_size)

            # 注意力机制参数
            self.num_heads = 4
            self.head_dim = dual_emb_size // self.num_heads
            assert self.head_dim * self.num_heads == dual_emb_size, "com_emb_size must be divisible by num_heads"

            # 多头注意力线性变换
            self.q_linear = nn.Linear(dual_emb_size, dual_emb_size)
            self.k_linear = nn.Linear(dual_emb_size, dual_emb_size)

            self.bn_instance = torch.nn.BatchNorm1d(dual_emb_size, affine=False)
            self.bn_user = torch.nn.BatchNorm1d(dual_emb_size, affine=False)
            self.single_weight = nn.Linear(20, 1, bias=False)

    def simple_dual_module(self, input):
        # 实例特征提取
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)
        instance_difficulty = F.normalize(instance_difficulty)

        # 用户特征提取
        user_feature = self.user_feature_1(self.user_feature_vec)
        user_feature = F.normalize(user_feature)

        batch_size = instance_difficulty.size(0)
        num_annotators = self.num_annotators

        # 多头注意力变换
        q = self.q_linear(instance_difficulty).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
        k = self.k_linear(user_feature).view(num_annotators, self.num_heads, self.head_dim).permute(1, 0, 2)  # [H, A, D]

        # 注意力计算
        attn_scores = torch.einsum('bhd,had->bha', q, k) / (self.head_dim ** 0.5)  # [B, H, A]
        combined_scores = attn_scores.mean(dim=1)  # 多头平均 [B, A]

        # 动态调整权重
        rectify_rate = torch.sigmoid(combined_scores)
        return rectify_rate

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
            cls_out = F.softmax(x, dim=1)

        if mode == 'train':
            #x = input.view(input.size(0), -1)
            instance_features = cls_out.view(cls_out.size(0), -1)

            if self.dual_module == 'simple':
                rectify_rate = self.simple_dual_module(instance_features)
            elif self.dual_module == 'gcn':
                u = list(range(self.num_annotators))
                rectify_rate, rec_out = self.gae(u, idx, support, support_t)
                rectify_rate = rectify_rate.transpose(0, 1)

            # 使用einsum操作计算共同概率
            instance_prob = torch.einsum('ij,jk->ik', (cls_out, self.instance_kernel))
            # 使用einsum操作计算个体概率
            indivi_prob = torch.einsum('ik,jkl->ijl', (cls_out, self.kernel))

            # 根据共同率组合共同概率和个体概率
            crowd_out = rectify_rate[:, :, None] * instance_prob[:, None, :] + (1 - rectify_rate[:, :, None]) * indivi_prob
            crowd_out = crowd_out.transpose(1, 2)

        if self.dual_module == 'simple' or mode == 'test':
            return cls_out, crowd_out