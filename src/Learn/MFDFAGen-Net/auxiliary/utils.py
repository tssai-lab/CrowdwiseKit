import os
import os.path
import copy
import hashlib  # 用于生成MD5哈希
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import torch.nn.functional as F


def check_integrity(fpath, md5):
    """检查文件完整性：验证文件的MD5哈希是否与给定值匹配"""
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # 分块读取文件（每次1MB），计算MD5
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False  # 哈希不匹配
    return True  # 文件完整


def download_url(url, root, filename, md5):
    """从指定URL下载文件，并验证完整性"""
    from six.moves import urllib

    root = os.path.expanduser(root)  # 展开用户目录（如 ~ -> /home/user）
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)  # 创建目录（如果不存在）
    except OSError as e:
        if e.errno == errno.EEXIST:  # 目录已存在则忽略
            pass
        else:
            raise

    # 如果文件存在且完整，直接使用
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:  # 否则下载
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':  # 尝试将https替换为http重试
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """列出目录下所有子目录"""
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),  # 过滤出目录
            os.listdir(root)
        )
    )
    # 是否返回完整路径
    return [os.path.join(root, d) for d in directories] if prefix else directories


def list_files(root, suffix, prefix=False):
    """列出目录下所有以指定后缀结尾的文件"""
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    # 是否返回完整路径
    return [os.path.join(root, d) for d in files] if prefix else files


# ---------------------------- 标签噪声生成函数 ----------------------------
def multiclass_noisify(y, P, random_state=0):
    """
    根据转移概率矩阵P生成多类别噪声标签
    参数:
        y: 原始标签数组
        P: 转移概率矩阵，P[i][j]表示类别i翻转到j的概率
        random_state: 随机种子
    返回:
        new_y: 含噪声的标签数组
    """
    assert P.shape[0] == P.shape[1]  # 确保P是方阵
    assert np.max(y) < P.shape[0]  # 标签值不超过类别数

    # 验证每行概率和为1
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()  # 概率非负

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)  # 固定随机种子

    # 对每个样本，根据P矩阵进行标签翻转
    for idx in np.arange(m):
        i = y[idx]
        # 按P[i]的概率分布进行多项式采样，确定新标签
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """
    生成相邻类别翻转噪声（Pairflip Noise）
    例如在CIFAR-10中，0→1→2→...→9→0形成一个环形翻转
    """
    P = np.eye(nb_classes)  # 初始为单位矩阵（无噪声）
    n = noise

    if n > 0.0:
        # 构建转移矩阵
        P[0, 0], P[0, 1] = 1. - n, n  # 类别0以概率n翻转到1
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n  # 中间类翻转到下一类
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n  # 最后一类翻转到0

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()  # 计算实际噪声率
        print(f'Actual noise: {actual_noise:.2f}')
        y_train = y_train_noisy
    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """生成对称均匀翻转噪声（Symmetric Noise）"""
    P = np.ones((nb_classes, nb_classes)) * (noise / (nb_classes - 1))  # 均匀分布噪声
    if noise > 0.0:
        np.fill_diagonal(P, 1. - noise)  # 对角线保持1-noise的概率

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        print(f'Actual noise: {actual_noise:.2f}')
        y_train = y_train_noisy
    return y_train, actual_noise


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    """噪声生成入口函数：根据类型调用具体噪声函数"""
    if noise_type == 'pairflip':
        return noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        return noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)


def noisify_instance(train_data, train_labels, noise_rate):
    """
    生成与实例相关的噪声（Feature-dependent Noise）
    通过样本特征和随机权重矩阵计算翻转概率
    """
    num_class = 100 if max(train_labels) > 10 else 10  # 判断是CIFAR-10还是100
    np.random.seed(0)

    # 生成噪声率分布（截断到0~1之间）
    q_ = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
    q = [pro for pro in q_ if 0 < pro < 1][:50000]  # 取前50000个有效值

    # 生成随机权重矩阵（特征维度 × 类别数）
    w = np.random.normal(loc=0, scale=1, size=(32 * 32 * 3, num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample_flat = sample.flatten()
        # 计算样本与所有类别的相关性分数
        p_all = np.matmul(sample_flat, w)
        p_all[train_labels[i]] = -1000000  # 排除真实类别
        # 计算翻转概率（使用Softmax归一化）
        p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]  # 真实类别的保持概率
        # 根据概率采样新标签
        noisy_labels.append(np.random.choice(num_class, p=p_all / p_all.sum()))

    # 计算总体噪声率
    over_all_noise_rate = 1 - (torch.tensor(train_labels) == torch.tensor(noisy_labels)).float().mean()
    return noisy_labels, over_all_noise_rate