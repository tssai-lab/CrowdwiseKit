import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
import datetime

import scipy.io as io
import csv
from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    #获得四个返回值，分别是support set的样本、support set的标签、query set的样本、query set的标签
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    # mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
    #                          k_query=args.k_qry,
    #                          batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time 60000/10000
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                #Mean validation accuracy/loss, stddev, and confidence intervals
                print('step:', step, '\ttraining acc:', accs)
            #过500个epoch，会将参数放到meta-test set上进行测试，然后对测试集中的每个task做fine-tune

    torch.save(maml,'pre_net3.pth')

def fine_tune():
    device = torch.device('cuda')
    model=torch.load('pre_net3.pth').to(device)

    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=200,
                             batchsz=1, resize=args.imgsz)

    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        feature = model.finetunning(x_spt, y_spt, x_qry, y_qry)


    # fx = open('data/dataset.csv', 'w', newline="")
    # csv_write = csv.writer(fx)
    # csv_write.writerows(x_qry)
    #
    # y_qry = y_qry.data.cpu().numpy()
    # # print(y_qry)
    # truth =[]
    # id=1
    # for x in y_qry:
    #     truth.append([id,x+1])
    #     id = id +1
    # f=open('data/mini_truth3.glod','w')
    # for i in range(len(truth)):
    #     f.write(' '.join('%s'% k for k in truth[i])+'\n')
    #
    # feature = feature.data.cpu().numpy()
    # # print(feature)
    # features=[]
    # id=1
    # for y in feature:
    #     y=np.insert(y,0,id)
    #     y=y.tolist()
    #     features.append(y)
    #     id = id +1
    # fp=open('data/mini_feature3.attr','w')
    # for j in range(len(features)):
    #     fp.write(' '.join('%s'% k for k in features[j])+'\n')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=30)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)  #对每个task进行gradient descent的次数
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1)

    args = argparser.parse_args()

    start = datetime.datetime.now()
    main()
    fine_tune()
    end = datetime.datetime.now()
    print("time:"+str((end-start).seconds/60)+"min")