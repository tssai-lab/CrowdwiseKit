"""
data - to generate data from crowds
"""
import pickle
import sys
import torch.utils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils_cifar10n import *


class CIFAR10N(Dataset):
    """
    to generate a dataset with images, experts' predictions and true labels for learning from crowds settings
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, data_path, mode="train", transform=None, download=False, sideinfo_path=None, split_ratio=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        if self.mode == 'test':
            f = self.test_list[0][0]
            file = os.path.join(self.data_path, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            test_data = entry['data']
            if 'labels' in entry:
                test_labels = entry['labels']
            else:
                test_labels = entry['fine_labels']
            fo.close()
            test_data = test_data.reshape((10000, 3, 32, 32))
            test_data = test_data.transpose((0, 2, 3, 1))

            self.data = test_data
            self.target_gt = test_labels
        else:
            '''get the data and ground-truth label'''
            train_data = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.data_path, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                train_data.append(entry['data'])
                fo.close()
            train_data = np.concatenate(train_data)
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            all_data = train_data
            


            data, target_gt, target_cs, target_cs_1hot = prep_cifar10n(all_data, sideinfo_path, split_ratio, mode=self.mode)

            self.data = data
            self.target_gt = target_gt
            self.target_cs = target_cs if self.mode == 'train' else None
            self.target_cs_1hot = target_cs_1hot if self.mode == 'train' else None
        



    def __getitem__(self, index):
        if self.mode == 'train':
            '''data, target_cs, target_cs_1hot, target_warmup, target_gt, index'''
            data = self.data[index]
            data = Image.fromarray(data)
            if self.transform is not None:
                data = self.transform(data)
            target_cs = self.target_cs[index]
            target_cs_1hot = self.target_cs_1hot[index]
            target_gt = self.target_gt[index]
            return data, target_cs, target_cs_1hot, target_gt, index

        else:
            data = self.data[index]
            data = Image.fromarray(data)
            if self.transform is not None:
                data = self.transform(data)
            target_gt = self.target_gt[index]
            return data, target_gt, index

    def __len__(self):
        return self.data.shape[0]

    def _check_integrity(self):
        root = self.data_path
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.data_path
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)



class CIFAR10NMETA(CIFAR10N):
    """
        Meta set Build for CIFAR
    """
    def __init__(self, meta_idx, meta_label, **kwargs):
        # drop_id must be a list containing index to drop
        super().__init__(**kwargs)
        self.data = self.data[meta_idx]
        self.targets = meta_label
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index):
        if self.mode == 'train':
            '''data, target_cs, target_cs_1hot, target_warmup, target_gt, index'''
            data = self.data[index]
            data = Image.fromarray(data)
            if self.transform is not None:
                data = self.transform(data)
            target = self.targets[index]
            return data, target


class CIFAR10N_PC(CIFAR10N):
    """
        distilled Build for CIFAR
    """
    def __init__(self, data_idx, cls, **kwargs):
        # drop_id must be a list containing index to drop
        super().__init__(**kwargs)
        self.data_idx = data_idx
        self.data = self.data[data_idx]
        self.targets = np.array([cls] * len(data_idx))
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index):
        if self.mode == 'train':
            '''data, target_cs, target_cs_1hot, target_warmup, target_gt, index'''

            data = self.data[index]
            data = Image.fromarray(data)
            if self.transform is not None:
                data = self.transform(data)
            target = self.targets[index]
            data_idx = self.data_idx[index]
            return data, target, data_idx



    
