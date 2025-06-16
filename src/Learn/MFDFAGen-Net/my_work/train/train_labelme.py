# 这是一个示例 Python 脚本。
import argparse
import random

from torch import optim
from torch.utils.data import DataLoader

from options import model_opts

from conal import CoNAL
from utils import Dataset
from workflow import *
from copy import deepcopy

#设置随机种子，可重复
seed = 11
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = 'labelme'
model_dir = '../model/'

#创建训练集加载器
train_dataset = Dataset(mode = 'train',dataset = dataset,sparsity=0)
train_loader = DataLoader(dataset=train_dataset,batch_size=1024,shuffle=True)

test_dataset = Dataset(mode = "test",dataset=dataset)
test_loader = DataLoader(dataset = test_dataset,batch_size=32,shuffle= True)

valid_dataset = Dataset(mode = 'valid',dataset= dataset)
valid_loader = DataLoader(dataset = valid_dataset,batch_size=32,shuffle=True)


def main(opt,model = None):
    train_acc_list = []
    test_acc_list = []

    #创建用户特征矩阵
    user_feature = np.eye(train_dataset.num_users)

    #加载或者创建模型
    # model = CoNAL(num_annotators=train_dataset.num_users, num_class=train_dataset.num_classes,
    #               input_dims=train_dataset.input_dims,
    #               user_feature=user_feature, gumbel_common = False).cuda()
    if model != None:
        model = model
    else:
        model = CoNAL(num_annotators=train_dataset.num_users, num_class=train_dataset.num_classes,common_module='simple',
                      input_dims=train_dataset.input_dims,input=train_dataset.X,
                      user_feature=user_feature).cuda()
    #初始化最佳验证准确率和最佳模型
    best_valid_acc = 0
    best_model = None
    lr = 1e-2

    #设置循环
    for epoch in range(opt.num_epochs):
        optimizer = optim.Adam(params=model.parameters(),lr = lr)
        train_acc = train(train_loader = train_loader,model = model,optimizer = optimizer,criterion=multi_loss, train_mode = 'simple')

        valid_acc, valid_f1, _ = test(model = model,test_loader=valid_loader)
        test_acc , test_f1, _  = test(model = model,test_loader= test_loader)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        #更新最新模型和最佳验证准确率
        if valid_acc >best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)
            #torch.save(best_model,"./model/best_seed2_%s_simple_model.pth"%dataset)

        #打印当前轮次的验证和测试准确率以及F1分数
        print("Epoch [%3d],valid acc :%.5f, Valid f1:%.5f"%(epoch,valid_acc,valid_f1))
        print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))


    #使用最佳模型在验证集上进行最终评估
    test_acc ,test_f1, _ = test(model = best_model,test_loader=test_loader)
    print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))
    return best_model, test_acc


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_opts(parser)
    opt = parser.parse_args()#解析命令行参数

    test_acc = []
    # model = torch.load(model_dir + 'best_attn_%s_model.pth'%dataset)

    _, acc = main(opt,model =None)
    test_acc.append(acc)
