import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics import euclidean_distances
from torch import Tensor
from torch.utils.data import Dataset

from utils import transform_onehot


class CustomDataset(Dataset):
    def __init__(self, mode='train', k=0, dataset='cifar10N', sparsity=0, test_ratio=0.0, transform=None):

        super(CustomDataset, self).__init__()
        self.mode = mode[:5] if mode.startswith('train') else mode
        self.transform = transform
        self.test_ratio = test_ratio

        X = None  # 初始化 X 为 None
        y = None  # 初始化 y 为 None

        if dataset == 'cifar10N':  # 如果数据集为labelme数据集
            data_path = '../data/cifar10N/raw'  # 设置数据路径
            # X = np.load(data_path + self.mode + '/data_%s.npy' % self.mode)  # 加载特征数据
            # print("x的形状：", X.shape)
            #
            # y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)  # 加载标签数据
            # print("y的形状：", y.shape)
            # X = X.reshape(X.shape[0], -1)  # 将特征数据重塑为2D数组
            # print("x reshape后的形状：", X.shape)
            X,y = self._load_cifar10_data(data_path, train=True)
            if mode == 'train':  # 如果是训练模式
                answers = np.load(data_path  + '/answers.npy')  # 加载多标注者答案
                print("answers的形状：", answers.shape)

                self.answers = answers
                self.num_users = answers.shape[1]  # 标注者的数量
                classes = np.unique(answers)  # 所有类别的集合
                if -1 in classes:  # 检查是否有缺失值
                    self.num_classes = len(classes) - 1  # 计算类别数（不包括缺失值）
                else:
                    self.num_classes = len(classes)
                input = X
                self.input_dims = input.view(input.size(0), -1).shape[1]  # 输入特征的维度
                self.answers_onehot = transform_onehot(answers, answers.shape[1], N_CLASSES=self.num_classes)  # 转换答案为one-hot编码
                print("one-hot形状", self.answers_onehot.shape)
            elif mode == 'train_dmi':  # 特殊训练模式
                answers = np.load(data_path + self.mode + '/answers.npy')  # 加载多标注者答案
                self.answers = transform_onehot(answers, answers.shape[1], N_CLASSES=self.num_classes)  # 转换答案为one-hot编码
                self.num_users = answers.shape[1]  # 标注者的数量
                classes = np.unique(answers)  # 所有类别的集合
                if -1 in classes:  # 检查是否有缺失值
                    self.num_classes = len(classes) - 1  # 计算类别数（不包括缺失值）
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]  # 输入特征的维度

        train_num = int(len(X) * (1 - test_ratio))  # 计算训练集样本数
        print("train_num:", train_num)

        self.X = X.float()[:train_num]  # 转换特征数据为PyTorch张量，并截取训练集部分
        self.X_val = X.float()[train_num:]  # 截取验证集部分

        if k:  # 如果指定了k值
            dist_mat = euclidean_distances(X, X)  # 计算特征数据间的欧氏距离矩阵
            k_neighbors = np.argsort(dist_mat, 1)[:, :k]  # 计算每个样本的k个最近邻居
            self.ins_feat = torch.from_numpy(X)  # 保存原始特征数据
            self.k_neighbors = k_neighbors  # 保存k个最近邻居
        self.y = torch.from_numpy(y)[:train_num]  # 转换标签数据为PyTorch张量，并截取训练集部分
        self.y_val = torch.from_numpy(y)[train_num:]  # 截取验证集部分
        if mode == 'train':  # 如果是训练模式
            self.ans_val = answers[train_num:]  # 截取验证集的答案部分

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
        print("X",X.shape)
        # 应用数据增强和标准化
        processed_images = []
        for img in X:
            pil_img = Image.fromarray(img)
            if self.transform:
                processed_img = self.transform(pil_img)
            else:
                processed_img = transforms.ToTensor()(pil_img)
            processed_images.append(processed_img)
        X = torch.stack(processed_images)
        print("after X",X.shape)


        return X, y

    # 保留原有其他方法...
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == 'train':  # 如果是训练模式
            return idx, self.X[idx], self.answers[idx], self.answers_onehot[idx], self.y[
                idx]  # 返回索引、特征、答案、one-hot编码答案和标签
        else:  # 如果不是训练模式
            return idx, self.X[idx], self.y[idx]  # 只返回索引、特征和标签


# 使用方法示例
if __name__ == '__main__':
    # 定义数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 创建数据集实例
    train_dataset = CustomDataset(
        dataset='cifar10n',
        mode='train',
        test_ratio=0.2,
        transform=train_transform
    )

    test_dataset = CustomDataset(
        dataset='cifar10n',
        mode='test',
        transform=test_transform
    )

    print(f"训练集尺寸: {len(train_dataset)}")
    print(f"测试集尺寸: {len(test_dataset)}")
    print(f"输入维度: {train_dataset.input_dims}")
    print(f"标注者数量: {train_dataset.num_users}")
    print(f"类别数量: {train_dataset.num_classes}")