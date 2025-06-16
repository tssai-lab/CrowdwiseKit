from cifar_dataset import CustomDataset
from utils import *
from torch import optim
from copy import deepcopy
import argparse
from options import *
from torch.utils.data import DataLoader
from utils import *
import random
from MFDFA_cifar10N import *
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score

seed = 100
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = 'cifar10N'
model_dir = '../model/'

# 定义数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = CustomDataset(mode='train', dataset=dataset, sparsity=0,test_ratio=0.2)
trn_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

valid_dataset = CustomDataset(mode='test', dataset=dataset)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(mode='test', dataset=dataset)
tst_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def main(opt, model=None):
    train_acc_list = []
    test_acc_list = []

    user_feature = np.eye(train_dataset.num_users)
    if model == None:
        model = MFDFA_cifar10N(num_annotators=train_dataset.num_users, num_classes=train_dataset.num_classes,
                               backbone_model="resnet110",
                            input_dims=train_dataset.input_dims, user_features=user_feature, use_gumbel=False).cuda()
    else:
        model = torch.load(model_dir + 'model%s' % dataset)

    best_valid_acc = 0
    best_model = None
    lr = 1e-2
    for epoch in range(100):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_acc = train(train_loader=trn_loader, model=model, optimizer=optimizer, criterion=multi_loss, train_mode='simple')
        valid_acc, valid_f1, _ = test(model=model, test_loader=val_loader)
        test_acc, test_f1, _ = test(model=model, test_loader=tst_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)

        print('Epoch [%3d], Valid acc: %.5f, Valid f1: %.5f' % (epoch, valid_acc, valid_f1))
        print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))

    test_acc, test_f1, _ = test(model=best_model, test_loader=tst_loader)
    print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))
    return best_model, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_opts(parser)
    opt = parser.parse_args()

    test_acc = []
    _, acc = main(opt, model=None)
    test_acc.append(acc)
