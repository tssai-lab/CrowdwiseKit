from __future__ import print_function  # 确保代码在 Python 2 和 Python 3 中兼容
from PIL import Image  # 用于图像处理
import os  # 用于文件路径操作
import os.path  # 用于文件路径操作
import numpy as np  # 用于数值计算
import sys  # 用于系统相关操作
if sys.version_info[0] == 2:  # 判断 Python 版本
    import cPickle as pickle  # Python 2 中的 pickle 模块
else:
    import pickle  # Python 3 中的 pickle 模块
import torch  # PyTorch 深度学习框架
import torch.utils.data as data  # PyTorch 数据加载工具
from .utils import download_url, check_integrity, multiclass_noisify  # 从 utils.py 导入工具函数

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): 数据集根目录。
        train (bool, optional): 如果为 True，加载训练集；否则加载测试集。
        transform (callable, optional): 图像变换函数。
        target_transform (callable, optional): 标签变换函数。
        download (bool, optional): 如果为 True，下载数据集。
        noise_type (str, optional): 噪声类型（如 'clean' 或 'human'）。
        noise_path (str, optional): 噪声标签文件路径。
        is_human (bool, optional): 是否使用人类标注的噪声标签。
    """
    base_folder = 'cifar-10-batches-py'  # CIFAR-10 数据集的文件夹名称
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"  # 数据集下载链接
    filename = "cifar-10-python.tar.gz"  # 数据集压缩文件名
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'  # 文件 MD5 校验码
    train_list = [  # 训练集文件列表（文件名和 MD5 校验码）
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [  # 测试集文件列表
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_path=None, is_human=True):
        self.root = "F:/research/dataset/cifar_N"  # 数据集根目录
        self.transform = transform  # 图像变换函数
        self.target_transform = target_transform  # 标签变换函数
        self.train = train  # 是否为训练集
        self.dataset = 'cifar10'  # 数据集名称
        self.noise_type = noise_type  # 噪声类型
        self.nb_classes = 10  # 类别数
        self.noise_path = noise_path  # 噪声标签文件路径
        idx_each_class_noisy = [[] for i in range(10)]  # 每个类别的噪声样本索引

        if download:  # 如果需要下载数据集
            self.download()

        # 加载数据
        if self.train:  # 加载训练集
            self.train_data = []  # 训练图像数据
            self.train_labels = []  # 训练标签
            for fentry in self.train_list:  # 遍历训练集文件
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:  # 根据 Python 版本加载数据
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])  # 加载图像数据
                if 'labels' in entry:  # 加载标签
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)  # 合并数据
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))  # 调整形状
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # 转换为 HWC 格式

            if noise_type != 'clean':  # 如果存在噪声标签
                train_noisy_labels = self.load_label()  # 加载噪声标签
                self.train_noisy_labels = train_noisy_labels.tolist()  # 转换为列表
                print(f'noisy labels loaded from {self.noise_path}')

                if not is_human:  # 如果不是人类标注的噪声标签
                    T = np.zeros((self.nb_classes, self.nb_classes))  # 初始化噪声转移矩阵
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T / np.sum(T, axis=1)  # 归一化
                    print(f'Noise transition matrix is \n{T}')
                    train_noisy_labels = multiclass_noisify(y=np.array(self.train_labels), P=T,
                                                           random_state=0)  # 生成合成噪声标签
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    T = np.zeros((self.nb_classes, self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T / np.sum(T, axis=1)
                    print(f'New synthetic noise transition matrix is \n{T}')

                for i in range(len(self.train_noisy_labels)):  # 统计每个类别的噪声样本
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)  # 计算噪声比例
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                self.actual_noise_rate = np.sum(self.noise_or_not) / 50000  # 计算总体噪声率
                print('over all noise rate is ', self.actual_noise_rate)
        else:  # 加载测试集
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']  # 加载测试图像数据
            if 'labels' in entry:  # 加载测试标签
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))  # 调整形状
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # 转换为 HWC 格式

    def load_label(self):
        # 加载噪声标签
        try:
            # 加载完整的多标注者答案
            all_answers = np.load(self.noise_path)
            assert all_answers.shape == (50000, 747)

            # 随机选择一个标注者的答案作为噪声标签
            np.random.seed(42)  # 保持可重复性
            selected_annotator = np.random.choice(747, size=50000)
            noise_label = all_answers[np.arange(50000), selected_annotator]

            return torch.from_numpy(noise_label)
        except Exception as e:
            print(f"Error loading noise labels: {e}")
            raise

    def __getitem__(self, index):
        """
        获取数据集中的单个样本。

        Args:
            index (int): 样本索引。

        Returns:
            tuple: (image, target, index) 其中 target 是目标类别索引。
        """
        if self.train:  # 如果是训练集
            if self.noise_type != 'clean':  # 如果存在噪声标签
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:  # 如果是测试集
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像

        if self.transform is not None:  # 应用图像变换
            img = self.transform(img)

        if self.target_transform is not None:  # 应用标签变换
            target = self.target_transform(target)

        return img, target, index  # 返回图像、标签和索引

    def __len__(self):
        """返回数据集的大小。"""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        """检查数据集文件的完整性。"""
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """下载数据集。"""
        import tarfile

        if self._check_integrity():  # 如果数据集已下载且完整
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)  # 下载数据集

        # 解压文件
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        """返回数据集的字符串表示。"""
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): 数据集根目录。
        train (bool, optional): 如果为 True，加载训练集；否则加载测试集。
        transform (callable, optional): 图像变换函数。
        target_transform (callable, optional): 标签变换函数。
        download (bool, optional): 如果为 True，下载数据集。
        noise_type (str, optional): 噪声类型（如 'clean' 或 'human'）。
        noise_rate (float, optional): 噪声率。
        random_state (int, optional): 随机种子。
        noise_path (str, optional): 噪声标签文件路径。
        is_human (bool, optional): 是否使用人类标注的噪声标签。
    """
    base_folder = 'cifar-100-python'  # CIFAR-100 数据集的文件夹名称
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"  # 数据集下载链接
    filename = "cifar-100-python.tar.gz"  # 数据集压缩文件名
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'  # 文件 MD5 校验码
    train_list = [  # 训练集文件列表
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [  # 测试集文件列表
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0, noise_path=None, is_human=True):
        super(CIFAR100, self).__init__(root, train, transform, target_transform, download,
                                      noise_type, noise_path, is_human)  # 调用父类构造函数
        self.nb_classes = 100  # 类别数
        idx_each_class_noisy = [[] for i in range(100)]  # 每个类别的噪声样本索引