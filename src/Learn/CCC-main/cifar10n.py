
"""
data - to generate data from crowds
"""
import pickle  # 用于加载保存为pickle格式的数据文件。
import sys  # 提供对一些与Python解释器直接交互的变量和函数的访问。
import torch.utils  # 包含构建数据加载器和其他工具函数。
from PIL import Image  # Python Imaging Library，用于打开、操作和保存多种图像文件格式。
from torch.utils.data import Dataset, DataLoader  # PyTorch中用于创建自定义数据集和加载器的基础类。
import numpy as np  # 用于数值计算的库。
from utils_cifar10n import *  # 导入特定于CIFAR-10N数据集的辅助函数。


class CIFAR10N(Dataset):
    """
    to generate a dataset with images, experts' predictions and true labels for learning from crowds settings
    """
    base_folder = 'cifar-10-batches-py'  # 定义CIFAR-10数据文件夹名称。
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"  # 定义下载CIFAR-10数据集的URL。
    filename = "cifar-10-python.tar.gz"  # 下载文件的名称。
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'  # 定义下载文件的MD5校验值。
    train_list = [  # 定义训练数据批处理列表。
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],

    ]

    test_list = [  # 定义测试数据批处理列表。
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, data_path, mode="train", transform=None, download=True, sideinfo_path=None, split_ratio=None):
        '''
        初始化CIFAR10N数据集实例。

        参数:
            data_path (str): 数据存储路径。
            mode (str): 模式("train" 或 "test")。
            transform (callable, optional): 可调用对象，应用于每个样本。
            download (bool): 如果True且数据不存在，则下载数据。
            sideinfo_path (str, optional): 侧信息路径，用于额外的数据处理。
            split_ratio (float, optional): 训练集划分比例。
        '''
        self.data_path = data_path  # 设置数据路径。
        self.mode = mode  # 设置模式。
        self.transform = transform  # 设置转换方法。
        if download:  # 如果需要下载数据。
            self.download()  # 调用download方法下载数据。

        if not self._check_integrity():  # 检查数据完整性。
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')  # 如果数据丢失或损坏，抛出异常。

        if self.mode == 'test':  # 如果是测试模式。
            f = self.test_list[0][0]  # 获取测试数据文件名。
            file = os.path.join(self.data_path, self.base_folder, f)  # 构造文件路径。
            fo = open(file, 'rb')  # 打开文件以读取二进制数据。
            entry = pickle.load(fo, encoding='latin1') if sys.version_info[0] != 2 else pickle.load(fo)  # 根据Python版本加载数据。

            test_data = entry['data']  # 获取测试数据。
            print("test_data.shape:",test_data.shape)

            test_labels = entry.get('labels', entry.get('fine_labels'))  # 获取测试标签。
            print("test_labels.shape:", test_labels.shape)
            fo.close()  # 关闭文件。
            test_data = test_data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))  # 将数据重塑并转置为正确的形状。

            self.data = test_data  # 设置测试数据。
            self.target_gt = test_labels  # 设置真实标签。

        else:  # 如果是训练模式。
            '''get the data and ground-truth label'''
            train_data = []  # 初始化训练数据列表。
            for fentry in self.train_list:  # 遍历所有训练数据文件。
                f = fentry[0]  # 获取文件名。
                file = os.path.join(self.data_path, self.base_folder, f)  # 构造文件路径。
                fo = open(file, 'rb')  # 打开文件以读取二进制数据。
                entry = pickle.load(fo, encoding='latin1') if sys.version_info[0] != 2 else pickle.load(fo)  # 根据Python版本加载数据。

                train_data.append(entry['data'])  # 将数据添加到训练数据列表。

                fo.close()  # 关闭文件。

            train_data = np.concatenate(train_data).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))  # 将所有训练数据合并，并重塑和转置为正确形状。
            all_data = train_data  # 设置所有数据。

            # 使用辅助函数准备数据。
            data, target_gt, target_cs, target_cs_1hot = prep_cifar10n(all_data, sideinfo_path, split_ratio,
                                                                       mode=self.mode)

            self.data = data  # 设置训练数据。
            self.target_gt = target_gt  # 设置真实标签。
            self.target_cs = target_cs if self.mode == 'train' else None  # 设置专家预测（仅限训练模式）。
            self.target_cs_1hot = target_cs_1hot if self.mode == 'train' else None  # 设置专家预测的one-hot编码（仅限训练模式）。

    def __getitem__(self, index):
        '''
        获取给定索引的数据项。

        参数:
            index (int): 数据项的索引。

        返回:
            tuple: 包含图像数据、目标标签等的数据项。
        '''
        if self.mode == 'train':
            '''data, target_cs, target_cs_1hot, target_warmup, target_gt, index'''
            data = self.data[index]  # 获取指定索引的数据。
            data = Image.fromarray(data)  # 将numpy数组转换为PIL图像。
            if self.transform is not None:  # 如果存在转换方法。
                data = self.transform(data)  # 应用转换方法。
            target_cs = self.target_cs[index]  # 获取指定索引的专家预测。
            target_cs_1hot = self.target_cs_1hot[index]  # 获取指定索引的专家预测的one-hot编码。
            target_gt = self.target_gt[index]  # 获取指定索引的真实标签。
            return data, target_cs, target_cs_1hot, target_gt, index  # 返回数据项。
        else:
            data = self.data[index]  # 获取指定索引的数据。
            data = Image.fromarray(data)  # 将numpy数组转换为PIL图像。
            if self.transform is not None:  # 如果存在转换方法。
                data = self.transform(data)  # 应用转换方法。
            target_gt = self.target_gt[index]  # 获取指定索引的真实标签。
            return data, target_gt, index  # 返回数据项。

    def __len__(self):
        '''
        获取数据集长度。

        返回:
            int: 数据集中的样本数量。
        '''
        return self.data.shape[0]  # 返回数据样本的数量。

    def _check_integrity(self):
        '''
        检查数据集的完整性。

        返回:
            bool: 数据是否完整。
        '''
        root = self.data_path  # 获取数据根目录。
        for fentry in (self.train_list + self.test_list):  # 遍历训练和测试数据文件。
            filename, md5 = fentry[0], fentry[1]  # 获取文件名和MD5校验值。
            fpath = os.path.join(root, self.base_folder, filename)  # 构造文件路径。
            if not check_integrity(fpath, md5):  # 检查文件完整性。
                return False  # 如果文件不完整，返回False。
        return True  # 如果所有文件都完整，返回True。

    def download(self):
        '''
        下载数据集。
        '''
        import tarfile  # 导入tarfile模块用于解压tar文件。

        if self._check_integrity():  # 检查数据完整性。
            print('Files already downloaded and verified')  # 如果数据已下载并验证，则打印消息。
            return  # 结束方法。

        root = self.data_path  # 获取数据根目录。
        download_url(self.url, root, self.filename, self.tgz_md5)  # 下载数据集。

        # 解压文件。
        cwd = os.getcwd()  # 获取当前工作目录。
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")  # 打开压缩文件。
        os.chdir(root)  # 切换到数据根目录。
        tar.extractall()  # 解压所有文件。
        tar.close()  # 关闭压缩文件。
        os.chdir(cwd)  # 切换回原始工作目录。


class CIFAR10NMETA(CIFAR10N):
    """
        Meta set Build for CIFAR
    """

    def __init__(self, meta_idx, meta_label, **kwargs):
        '''
        初始化CIFAR10NMETA数据集实例。

        参数:
            meta_idx (list): 元数据索引。
            meta_label (list): 元数据标签。
            **kwargs: 其他关键字参数传递给父类构造函数。
        '''
        super().__init__(**kwargs)  # 调用父类构造函数。
        self.data = self.data[meta_idx]  # 根据元数据索引筛选数据。
        self.targets = meta_label  # 设置元数据标签。
        assert len(self.data) == len(self.targets)  # 确保数据和标签长度一致。

    def __getitem__(self, index):
        '''
        获取给定索引的数据项。

        参数:
            index (int): 数据项的索引。

        返回:
            tuple: 包含图像数据和目标标签的数据项。
        '''
        if self.mode == 'train':
            '''data, target_cs, target_cs_1hot, target_warmup, target_gt, index'''
            data = self.data[index]  # 获取指定索引的数据。
            data = Image.fromarray(data)  # 将numpy数组转换为PIL图像。
            if self.transform is not None:  # 如果存在转换方法。
                data = self.transform(data)  # 应用转换方法。
            target = self.targets[index]  # 获取指定索引的目标标签。
            return data, target  # 返回数据项。


class CIFAR10N_PC(CIFAR10N):
    """
        distilled Build for CIFAR
    """

    def __init__(self, data_idx, cls, **kwargs):
        '''
        初始化CIFAR10N_PC数据集实例。

        参数:
            data_idx (list): 数据索引。
            cls (list): 类别标签。
            **kwargs: 其他关键字参数传递给父类构造函数。
        '''
        super().__init__(**kwargs)  # 调用父类构造函数。
        self.data_idx = data_idx  # 设置数据索引。
        self.data = self.data[data_idx]  # 根据数据索引筛选数据。
        self.targets = np.array([cls] * len(data_idx))  # 设置类别标签。
        assert len(self.data) == len(self.targets)  # 确保数据和标签长度一致。

    def __getitem__(self, index):
        '''
        获取给定索引的数据项。

        参数:
            index (int): 数据项的索引。

        返回:
            tuple: 包含图像数据、目标标签和数据索引的数据项。
        '''
        if self.mode == 'train':
            '''data, target_cs, target_cs_1hot, target_warmup, target_gt, index'''

            data = self.data[index]  # 获取指定索引的数据。
            data = Image.fromarray(data)  # 将numpy数组转换为PIL图像。
            if self.transform is not None:  # 如果存在转换方法。
                data = self.transform(data)  # 应用转换方法。
            target = self.targets[index]  # 获取指定索引的目标标签。
            data_idx = self.data_idx[index]  # 获取指定索引的数据索引。
            return data, target, data_idx  # 返回数据项。
