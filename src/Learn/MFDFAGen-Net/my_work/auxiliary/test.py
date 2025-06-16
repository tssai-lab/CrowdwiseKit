# 导入所需的库和模块
from MFDFA import *  # 假设这是自定义的模块，用于深度学习模型的定义等
from utils import *  # 假设这是工具函数模块
from torch import optim  # 导入PyTorch的优化器模块
from copy import deepcopy  # 导入深度拷贝函数
import argparse  # 导入用于解析命令行参数的模块
from options import *  # 假设这是定义了一些配置选项的模块
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载器
from workflow import *  # 假设这是定义了训练和测试流程的模块
import random  # 导入随机数生成模块
from sklearn.decomposition import NMF  # 导入非负矩阵分解算法
from sklearn.metrics import accuracy_score  # 导入准确率计算函数

# 设置随机种子以保证结果的可重复性
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)  # 注意：这里假设已经导入了numpy并简称为np，但在代码中未显示

# 定义数据集名称和模型保存目录
dataset = 'labelme'
model_dir = '../model/'

# 创建验证和测试数据集及其数据加载器

valid_dataset = Dataset(mode='valid', dataset=dataset)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

test_dataset = Dataset(mode='test', dataset=dataset)
tst_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# 定义主函数
def main(opt):
    # 初始化训练和测试准确率列表
    train_acc_list = []
    test_acc_list = []


    # 加载或创建模型

    model = torch.load(model_dir + 'best_seed14_%s_simple_model.pth' % dataset)

    # 初始化最佳验证准确率和最佳模型
    best_valid_acc = 0
    best_model = None
    lr = 1e-2  # 设置学习率

    # 训练循环

    optimizer = optim.Adam(model.parameters(), lr=lr)  # 创建优化器

    valid_acc, valid_f1, _ = test(model=model, test_loader=val_loader)  # 在验证集上评估模型
    test_acc, test_f1, _ = test(model=model, test_loader=tst_loader)  # 在测试集上评估模型

    test_acc_list.append(test_acc)

    # 如果当前验证准确率高于最佳验证准确率，则更新最佳模型和最佳验证准确率
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = deepcopy(model)
        torch.save(best_model,"./model/best_seed14_%s_simple_model.pth"% dataset)

    # 打印当前轮次的验证和测试准确率及F1分数
    print(', Valid acc: %.5f, Valid f1: %.5f' % (valid_acc, valid_f1))
    print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))

    # 使用最佳模型在测试集上进行最终评估
    test_acc, test_f1, _ = test(model=best_model, test_loader=tst_loader)
    print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))
    return best_model, test_acc


# 如果直接运行此脚本，则执行以下代码
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    model_opts(parser)  # 假设这是向解析器添加模型相关选项的函数
    opt = parser.parse_args()  # 解析命令行参数

    test_acc = []  # 初始化测试准确率列表
    _, acc = main(opt)  # 调用主函数，这里model=True可能是个错误或特殊用法，通常应传入一个模型实例或None
    test_acc.append(acc)  # 将测试准确率添加到列表中