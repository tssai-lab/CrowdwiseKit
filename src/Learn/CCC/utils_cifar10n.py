"""
util - other functions
"""
import torch
import numpy as np
from torch.autograd import Variable
import csv
import os
import os.path
import hashlib
import errno



def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def convert_one_hot(label_cs):
    nb_classes = len(np.unique(label_cs)) - 1 
    label_cs_1hot = np.zeros((label_cs.shape[0], label_cs.shape[1], nb_classes))
    data_idx = np.where(label_cs != -1)[0]
    annot_idx = np.where(label_cs != -1)[1]


    for i in range(len(data_idx)):
        label_cs_1hot[data_idx[i]][annot_idx[i]] = np.eye(nb_classes)[int(label_cs[data_idx[i]][annot_idx[i]])]

    return label_cs_1hot



def prep_cifar10n(all_data, sideinfo_path, split_ratio, mode="train"):
    '''get target_cs, target_cs_1hot, target_warmup'''

    rows = []
    with open(os.path.join(sideinfo_path, 'side_info_cifar10N.csv')) as fo:
        csv_reader = csv.reader(fo)
        for row in csv_reader:
            rows.append(row)
        rows.pop(0)
    rows = np.array(rows)
    annot = rows[:, [1, 3, 5]].astype(int)  # the index of annotators  50000 x 3

    label_info = torch.load(os.path.join(sideinfo_path, 'CIFAR-10_human.pt'))
    random_info_1 = label_info['random_label1']
    random_info_2 = label_info['random_label2']
    random_info_3 = label_info['random_label3']

    label_true = label_info['clean_label']
    target_cs = -1 * np.ones((50000, 747), dtype=np.int64)

    for i in range(5000):
        for j in range(10):
            target_cs[10 * i + j][annot[i][0]] = random_info_1[10 * i + j]
            target_cs[10 * i + j][annot[i][1]] = random_info_2[10 * i + j]
            target_cs[10 * i + j][annot[i][2]] = random_info_3[10 * i + j]


    target_cs_1hot = convert_one_hot(target_cs)

    "split the training set and validation set"
    split_index = []
    nclass = len(np.unique(label_true))
    if mode == 'train':
        for i in range(nclass):
            split_index.append(np.where(label_true == i)[0][:int(split_ratio * len(all_data) / nclass)])
    elif mode == 'val':
        for i in range(nclass):
            split_index.append(np.where(label_true == i)[0][int(split_ratio * len(all_data) / nclass):])
    split_index = np.array(split_index).reshape(-1).astype('int')

    data = all_data[split_index]
    target_gt = label_true[split_index]
    target_cs = target_cs[split_index]
    target_cs_1hot = target_cs_1hot[split_index]


    return data, target_gt, target_cs, target_cs_1hot




