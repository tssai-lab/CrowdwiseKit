
from utils import *
from torch import optim
from copy import deepcopy
import argparse
from options import *
from torch.utils.data import DataLoader
from workflow import *
import random

from MFDFA_cifar10N import *
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = 'cifar10N'
model_dir = '../model/'

train_data = np.load('../data/cifar10N/train/data_train.npy')
print(train_data.shape)
print(train_data[:10])

music_data = np.load('../data/music/train/data_train.npy')
print(music_data.shape)
print(music_data[:10])