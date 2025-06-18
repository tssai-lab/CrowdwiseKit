import numpy as np  # 导入NumPy库，用于高效的数组和矩阵运算
from torch.utils import data  # 从PyTorch导入data模块，用于数据加载
import torch  # 导入PyTorch库
from sklearn.metrics.pairwise import euclidean_distances  # 导入计算欧式距离的函数
from sklearn.model_selection import train_test_split  # 导入用于分割数据集的函数
import IPython  # 导入IPython库，通常用于交互式编程
import os  # 导入os模块，用于操作系统相关的功能
from torchvision.datasets.utils import download_url, check_integrity  # 导入用于下载和验证数据完整性的函数
import torchvision.transforms as transforms  # 导入PyTorch的图像预处理模块
import sys  # 导入sys模块，用于访问与Python解释器相关的变量和函数
import pandas as pd  # 导入Pandas库，用于数据处理和分析
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import torchvision.models as models  # 导入PyTorch的预训练模型
from sklearn.preprocessing import normalize  # 导入用于数据归一化的函数

from auxiliary.cifar import CIFAR10


def map_data(data):
    """
    将数据映射到连续的索引范围内，如果数据本身不是连续的。

    参数:
    data : np.int32 数组

    返回:
    mapped_data : 映射后的数据数组
    id_dict : 原始值到新值的字典映射
    n : 映射后数据的长度
    """
    uniq = list(set(data))  # 获取所有唯一值
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}  # 创建从旧值到新值的映射
    data = np.array(list(map(lambda x: id_dict[x], data)))  # 应用映射转换
    n = len(uniq)  # 计算唯一值的数量

    return data, id_dict, n

def one_hot(target, n_classes):
    """
    将标签转换为one-hot编码形式。

    参数:
    target : 目标标签
    n_classes : 类别总数

    返回:
    one_hot_targets : one-hot编码的标签
    """
    targets = np.array([target]).reshape(-1)  # 转换为目标标签数组
    one_hot_targets = np.eye(n_classes)[targets]  # 使用numpy eye函数创建one-hot编码矩阵
    return one_hot_targets

def transform_onehot(answers, N_ANNOT, N_CLASSES, empty=-1):
    """
    将带有缺失值的多标注者答案转换为one-hot编码形式。

    参数:
    answers : 多标注者答案数组
    N_ANNOT : 标注者的数量
    N_CLASSES : 类别总数
    empty : 缺失值标记，默认为-1

    返回:
    answers_bin_missings : 包含缺失值的one-hot编码答案数组
    """
    answers_bin_missings = []
    for i in range(len(answers)):  # 遍历每个样本的答案
        row = []
        for r in range(N_ANNOT):  # 遍历每个标注者
            if answers[i, r] == -1:  # 如果答案是缺失的
                row.append(empty * np.ones(N_CLASSES))  # 添加一个全为empty的数组
            else:
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])  # 否则添加one-hot编码的答案
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)  # 调整维度顺序
    return answers_bin_missings

class Dataset(data.Dataset):
    """
    定义一个PyTorch数据集类，用于处理训练和测试数据。
    """
    def __init__(self, mode='train', k=0, dataset='labelme', sparsity=0, test_ratio=0,transform = None):
        """
        初始化数据集类。

        参数:
        mode : 数据集模式（如'train'或'test'）
        k : 用于计算最近邻的数量
        dataset : 数据集名称（如'labelme'或'music'）
        sparsity : 稀疏性参数，未在代码中使用
        test_ratio : 测试集占总数据的比例
        """
        self.transform = transform
        if mode[:5] == 'train':  # 判断是否为训练模式
            self.mode = mode[:5]
        else:
            self.mode = mode

        if dataset == 'music':  # 如果数据集为音乐数据集
            data_path = '../data/music/'  # 设置数据路径
            X = np.load(data_path + self.mode + '/data_%s.npy' % self.mode)
            y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)
            if mode == 'train':
                answers = np.load(data_path + self.mode +'/answers.npy')
                self.answers = answers
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)

                self.input_dims = X.shape[1]  # 输入特征的维度
                self.answers_onehot = transform_onehot(answers, answers.shape[1], self.num_classes)  # 转换答案为one-hot编码

        elif dataset == 'labelme':
            data_path = '../data/labelme/'  # 设置数据路径
            X = np.load(data_path + self.mode + '/data_%s_vgg16.npy' % self.mode)  # 加载特征数据


            y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)  # 加载标签数据

            X = X.reshape(X.shape[0], -1)  # 将特征数据重塑为2D数组

            if mode == 'train':  # 如果是训练模式
                answers = np.load(data_path + self.mode + '/answers.npy')  # 加载多标注者答案


                self.answers = answers
                self.num_users = answers.shape[1]  # 标注者的数量
                classes = np.unique(answers)  # 所有类别的集合
                if -1 in classes:  # 检查是否有缺失值
                    self.num_classes = len(classes) - 1  # 计算类别数（不包括缺失值）
                else:
                    self.num_classes = len(classes)

                #self.input_dims = self.num_classes
                self.input_dims = X.shape[1]  # 输入特征的维度
                self.answers_onehot = transform_onehot(answers, answers.shape[1], 8)  # 转换答案为one-hot编码

            elif mode == 'train_dmi':  # 特殊训练模式
                answers = np.load(data_path + self.mode + '/answers.npy')  # 加载多标注者答案
                self.answers = transform_onehot(answers, answers.shape[1], 8)  # 转换答案为one-hot编码
                self.num_users = answers.shape[1]  # 标注者的数量
                classes = np.unique(answers)  # 所有类别的集合
                if -1 in classes:  # 检查是否有缺失值
                    self.num_classes = len(classes) - 1  # 计算类别数（不包括缺失值）
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]  # 输入特征的维度



        train_num = int(len(X) * (1 - test_ratio))  # 计算训练集样本数
        print("train_num:",train_num)

        self.X = torch.from_numpy(X).float()[:train_num]  # 转换特征数据为PyTorch张量，并截取训练集部分
        self.X_val = torch.from_numpy(X).float()[train_num:]  # 截取验证集部分

        if k:  # 如果指定了k值
            dist_mat = euclidean_distances(X, X)  # 计算特征数据间的欧氏距离矩阵
            k_neighbors = np.argsort(dist_mat, 1)[:, :k]  # 计算每个样本的k个最近邻居
            self.ins_feat = torch.from_numpy(X)  # 保存原始特征数据
            self.k_neighbors = k_neighbors  # 保存k个最近邻居
        self.y = torch.from_numpy(y)[:train_num]  # 转换标签数据为PyTorch张量，并截取训练集部分
        self.y_val = torch.from_numpy(y)[train_num:]  # 截取验证集部分
        if mode == 'train':  # 如果是训练模式
            self.ans_val = answers[train_num:]  # 截取验证集的答案部分

        # 新增CIFAR10数据加载方法

    def _load_cifar10_data(self, data_path, train=True):
        """加载CIFAR-10原始数据并进行预处理"""
        # 确定要加载的文件
        filenames = [
            'data_batch_1', 'data_batch_2', 'data_batch_3',
            'data_batch_4', 'data_batch_5'
        ] if train else ['test_batch']

        images, labels = [], []
        for filename in filenames:
            filepath = os.path.join(data_path, filename)
            with open(filepath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                images.append(entry['data'])
                labels.extend(entry['labels'])

        # 合并数据并转换格式
        X = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC格式
        y = np.array(labels, dtype=np.int64)

        # 应用数据增强和标准化
        processed_images = []
        for img in X:
            pil_img = Image.fromarray(img)
            if self.transform:
                processed_img = self.transform(pil_img)
            else:
                processed_img = transforms.ToTensor()(pil_img)
            processed_images.append(processed_img)

        return torch.stack(processed_images), y



    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项。

        参数:
        idx : 索引

        返回:
        根据数据集模式返回不同的数据项。
        """
        if self.mode == 'train':  # 如果是训练模式
            return idx, self.X[idx], self.answers[idx], self.answers_onehot[idx], self.y[idx]  # 返回索引、特征、答案、one-hot编码答案和标签
        else:  # 如果不是训练模式
            return idx, self.X[idx], self.y[idx]  # 只返回索引、特征和标签

def simple_majority_voting(response, empty=-1):
    """
    实现简单的多数投票机制来处理多标注者答案。

    参数:
    response : 多标注者答案数组
    empty : 缺失值标记，默认为-1

    返回:
    mv : 多数投票后的最终答案
    """
    mv = []
    for row in response:  # 遍历每个样本的答案
        bincount = np.bincount(row[row != empty])  # 统计非空答案出现次数
        mv.append(np.argmax(bincount))  # 选择出现次数最多的答案作为最终答案
    return np.array(mv)  # 返回多数投票结果




#工作流

from loss import *
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import IPython

# 定义交叉熵损失函数，并将它移动到GPU上（如果可用）
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').cuda()


def train(train_loader, model, optimizer, criterion=F.cross_entropy, train_mode='simple', annotators=None, pretrain=None,
          support=None, support_t=None, scale=0):
    """
    训练模型的函数。

    参数:
    train_loader : DataLoader对象，用于加载训练数据
    model : 要训练的PyTorch模型
    optimizer : 优化器
    criterion : 损失计算函数，默认为F.cross_entropy
    mode : 模型训练模式，可以是'simple'、'common'或其他模式
    annotators : 标注者信息，未在代码中使用
    pretrain : 预训练参数，未在代码中使用
    support : 支持集，未在代码中使用
    support_t : 支持集标签，未在代码中使用
    scale : 缩放因子，默认为0
    """

    model.train()  # 设置模型为训练模式
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    total_loss = 0  # 总损失
    loss = 0  # 单次迭代的损失

    correct_rec = 0  #
    total_rec = 0  #

    for idx, input, targets, targets_onehot, true_labels in train_loader:  # 遍历训练数据集
        # 将数据和标签移动到GPU（如果可用）
        input = input.cuda()
        targets = targets.cuda().long()
        targets_onehot = targets_onehot.cuda()
        targets_onehot[targets_onehot == -1] = 0  # 将-1标记的缺失值替换为0
        true_labels = true_labels.cuda().long()

        if train_mode == 'simple':  # 如果是简单模式
            loss = 0
            if scale:  # 如果scale不为0
                cls_out, output = model(input)
                #cls_out, output, trace_norm = model(input)  # 模型输出分类结果、原始输出和trace norm
                #loss += scale * trace_norm  # 增加正则化项
                mask = targets != -1  # 创建一个mask来忽略-1标记的缺失值

                y_pred = torch.transpose(output, 1, 2)  # 转置output以适应后续计算
                y_true = torch.transpose(targets_onehot, 1, 2).float()  # 转置并转换类型
                loss += torch.mean(-y_true[mask] * torch.log(y_pred[mask]))  # 计算损失并增加到总损失
            else:
                cls_out, output = model(input)  # 模型输出分类结果和原始输出

                loss += criterion(targets, output)  # 使用给定的criterion计算损失

            _, predicted = cls_out.max(1)  # 获取最大值的索引作为预测类别

            correct += predicted.eq(true_labels).sum().item()  # 统计正确预测的数量
            total += true_labels.size(0)  # 更新总样本数
        elif train_mode == 'common':  # 如果是common模式
            rec_loss = 0  # 初始化重构损失
            loss = 0  # 初始化损失
            cls_out, output = model(input, mode='train')  # 模型输出分类结果和原始输出
            _, predicted = cls_out.max(1)  # 获取最大值的索引作为预测类别
            correct += predicted.eq(true_labels).sum().item()  # 统计正确预测的数量
            total += true_labels.size(0)  # 更新总样本数
            loss += criterion(targets, output)  # 使用给定的criterion计算损失
            # print("output.shape:", output.shape,"targets.shape:",targets.shape)
            # 减去kernel与common_kernel之间差异的L2范数的和，作为正则化项
            loss -= 0.00001 * torch.sum(
                torch.norm((model.annotator_probs - model.instance_probs).view(targets.shape[1], -1), dim=1, p=2))
        else:  # 其他模式
            output, _ = model(input)  # 模型输出

            loss = loss_fn(output, true_labels)  # 使用预定义的loss_fn计算损失

            _, predicted = output.max(1)  # 获取最大值的索引作为预测类别
            correct += predicted.eq(true_labels).sum().item()  # 统计正确预测的数量
            total += true_labels.size(0)  # 更新总样本数

        total_loss += loss.item()  # 累加损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    if train_mode == 'simple' or train_mode == 'common':
        print('Training acc: ', correct / total)  # 打印训练准确率
        return correct / total  # 返回训练准确率


def test(model, test_loader):
    """
    测试模型的函数。

    参数:
    model : 要测试的PyTorch模型
    test_loader : DataLoader对象，用于加载测试数据
    """

    model.eval()  # 设置模型为评估模式
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    target = []  # 存储真实标签
    predict = []  # 存储预测结果

    with torch.no_grad():  # 关闭自动求导，节省内存和计算资源
        for _, inputs, targets in test_loader:  # 遍历测试数据集
            inputs = inputs.cuda()  # 将输入数据移动到GPU（如果可用）
            target.extend(targets.data.numpy())  # 将真实标签添加到列表中
            targets = targets.cuda()  # 将标签移动到GPU（如果可用）

            total += targets.size(0)  # 更新总样本数
            output, _ = model(inputs, mode='test')  # 模型输出
            _, predicted = output.max(1)  # 获取最大值的索引作为预测类别
            predict.extend(predicted.cpu().data.numpy())  # 将预测结果添加到列表中
            correct += predicted.eq(targets).sum().item()  # 统计正确预测的数量

    acc = correct / total
    f1 = f1_score(target, predict, average='macro')  # 计算F1分数

    classes = list(set(target))
    classes.sort()
    acc_per_class = []
    predict = np.array(predict)
    target = np.array(target)
    for i in range(len(classes)):
        instance_class = target == i
        acc_i = np.mean(predict[instance_class] == classes[i])
        acc_per_class.append(acc_i)

    return acc, f1, acc_per_class






