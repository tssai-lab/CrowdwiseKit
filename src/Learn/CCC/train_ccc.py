
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import numpy as np
import os
import time
import torch.nn as nn
from utils_ccc import get_model, get_transforms, get_optimizer_and_scheduler, get_metaset, get_dataset, \
    actual_step, meta_step, train_CL_1iter, eval, print_args, set_seed, get_perclass_sets, eval_mean, finetune
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from fast_pytorch_kmeans import KMeans
from tqdm import tqdm


def main(args):
    cudnn.benchmark = True
    print("CUDA CUDNN STATE: {}".format(cudnn.benchmark))

    dst_train, dst_test, dst_val, args.nclass, args.in_dim, args.nb_annotator = get_dataset(args)

    print("----------# dataset      # {:7s}  --------".format(args.dataset))
    print("----------# train size   # {:7d}  --------".format(dst_train.__len__()))
    print("----------# test size    # {:7d}  --------".format(dst_test.__len__()))
    print("----------# val size     # {:7d}  --------".format(dst_val.__len__()))
    print("----------# nclass       # {:7d}  --------".format(args.nclass))
    print("----------# in_dim       # {:7d}  --------".format(args.in_dim))
    print("----------# nb_annotator # {:7d}  --------".format(args.nb_annotator))

    loader_train = DataLoader(dst_train, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=True)
    loader_test = DataLoader(dst_test, batch_size=args.test_bs, pin_memory=True, num_workers=0, shuffle=False)
    loader_val = DataLoader(dst_val, batch_size=args.test_bs, pin_memory=True, num_workers=0, shuffle=False)

    pc_sets = get_perclass_sets(args, dst_train)



    

    model_1, model_2 = get_model(args.arch, args.nclass, args.in_dim, args.nb_annotator)
    model_1, model_2 = model_1.to(args.device), model_2.to(args.device)



    optimizer_1, scheduler_1 = get_optimizer_and_scheduler(args, model_1)
    optimizer_2, scheduler_2 = get_optimizer_and_scheduler(args, model_2)

    criterion = nn.CrossEntropyLoss().cuda()
    #

    warmup_model_path = os.path.join(args.results_dir, args.dataset, 'warmup_models.pth')

            
    '''start warming up the model'''
    if not args.no_warm and not os.path.exists(warmup_model_path):
        print("Warm-up using CrowdLayer Starting at ", str(time.asctime(time.localtime(time.time()))))
        best_acc_CL_1 = 0
        best_acc_CL_2 = 0
        warmup_model_sd = {"model_1": model_1.state_dict(), "model_2": model_2.state_dict()}
        for epoch in range(args.epoch_warmup):
            model_1.train()
            model_2.train()

            losses_1, accs_1 = [], []
            losses_2, accs_2 = [], []


            for i, batch in enumerate(loader_train):
                img, target_cs_1hot, target_gt = batch[0].to(args.device), batch[2].to(args.device).float(), \
                                                batch[3].to(args.device)# only need the img and 1hot-target and target_gt
                '''train model-1'''
                loss_1, acc_1 = train_CL_1iter(model_1, img, target_cs_1hot, target_gt, optimizer_1)
                losses_1.append(loss_1)
                accs_1.append(acc_1)
                '''train model-2'''
                loss_2, acc_2 = train_CL_1iter(model_2, img, target_cs_1hot, target_gt, optimizer_2)
                losses_2.append(loss_2)
                accs_2.append(acc_2)


            train_loss_1, train_acc_1 = np.mean(losses_1), np.mean(accs_1)
            train_loss_2, train_acc_2 = np.mean(losses_2), np.mean(accs_2)

            eval_loss_1, eval_loss_2, eval_acc_1, eval_acc_2 = eval(model_1, model_2, loader_val, criterion, args.device)

            print('Warmup Epoch: {:2d}/{:2d}\n'
                   'Train Loss  : {:.4f} | {:.4f}\n'
                   'Train Acc   : {:.2f} | {:.2f}\n'
                   'Val  Loss   : {:.4f} | {:.4f}\n'
                   'Val  Acc    : {:.2f} | {:.2f}'.format(epoch+1, args.epoch_warmup, train_loss_1, train_loss_2,\
                    train_acc_1, train_acc_2, eval_loss_1, eval_loss_2, eval_acc_1, eval_acc_2))

            
            if best_acc_CL_1 < eval_acc_1:
                best_acc_CL_1 = eval_acc_1
                warmup_model_sd['model_1'] = model_1.state_dict()
            if best_acc_CL_2 < eval_acc_2:
                best_acc_CL_2 = eval_acc_2
                warmup_model_sd['model_2'] = model_2.state_dict()

        torch.save(warmup_model_sd, warmup_model_path)
        print("Warmup Train Ended at ", time.asctime(time.localtime(time.time())))
        print("Best Warmup accuracy: {:.2f} | {:.2f}".format(best_acc_CL_1, best_acc_CL_2))
    elif not args.no_warm:
        model_1.load_state_dict(torch.load(warmup_model_path)['model_1'])
        model_2.load_state_dict(torch.load(warmup_model_path)['model_2'])

    '''start Meta Confusion Corrcetion'''
    print("Training with Coupled Confusion Correction, Starting at ", str(time.asctime(time.localtime(time.time()))))
    best_acc_CCC = 0


    eval_loss_1, eval_loss_2, eval_acc_1, eval_acc_2 = eval(model_1, model_2, loader_val, criterion, args.device)
    print('State after warming up:\n'
          'Val Loss   : {:.4f} | {:.4f}\n'
          'Val Acc    : {:.2f} | {:.2f}'.format(eval_loss_1, eval_loss_2, eval_acc_1, eval_acc_2))


    for ep in range(args.epoch_ccc):
        model_1.train()
        model_2.train()
        '''group all the annotators into args.nb_group groups using k-means'''
        with torch.no_grad():
            CM_1 = model_1.get_mw_params()
            CM_2 = model_2.get_mw_params()
            CM_1 = CM_1.view(CM_1.size(0), -1)
            CM_2 = CM_2.view(CM_2.size(0), -1)
            kmeans = KMeans(n_clusters=args.nb_group, mode='euclidean', verbose=1)
            group_label_1 = kmeans.fit_predict(CM_1).tolist()
            group_label_2 = kmeans.fit_predict(CM_2).tolist()
        '''done grouping'''


        metaset_1, metaset_2 = get_metaset(args, dst_train, model_1, model_2, pc_sets)
        meta_loader_1 = DataLoader(metaset_1, batch_size=args.meta_bs, shuffle=True, num_workers=args.num_workers)
        meta_loader_2 = DataLoader(metaset_2, batch_size=args.meta_bs, shuffle=True, num_workers=args.num_workers)
        meta_loader_iter_1 = iter(meta_loader_1)
        meta_loader_iter_2 = iter(meta_loader_2)

        meta_losses = [[], []]
        actual_losses = [[], []]
        accs = [[], []]
        for batch_i, train_batch in tqdm(enumerate(loader_train)):
            try:
                meta_batch_1 = next(meta_loader_iter_1)
                meta_batch_2 = next(meta_loader_iter_2)
            except StopIteration as e:
                meta_loader_iter_1 = iter(meta_loader_1)
                meta_loader_iter_2 = iter(meta_loader_2)
                meta_batch_1 = next(meta_loader_iter_1)
                meta_batch_2 = next(meta_loader_iter_2)

            '''train model-1'''
            meta_loss_1, CM_CCC_1 = meta_step(args, model_1, optimizer_1, train_batch, meta_batch_2, group_label_1)
            actual_loss_1, train_acc_1 = actual_step(args, model_1, optimizer_1, train_batch, CM_CCC_1)
            '''train model-2'''
            meta_loss_2, CM_CCC_2 = meta_step(args, model_2, optimizer_2, train_batch, meta_batch_1, group_label_2)
            actual_loss_2, train_acc_2 = actual_step(args, model_2, optimizer_2, train_batch, CM_CCC_2)
            
            meta_losses[0].append(meta_loss_1)
            meta_losses[1].append(meta_loss_2)
            actual_losses[0].append(actual_loss_1)
            actual_losses[1].append(actual_loss_2)
            accs[0].append(train_acc_1)
            accs[1].append(train_acc_2)
        meta_loss_1, actual_loss_1, train_acc_1 = np.mean(meta_losses[0]), np.mean(actual_losses[0]), np.mean(accs[0])
        meta_loss_2, actual_loss_2, train_acc_2 = np.mean(meta_losses[1]), np.mean(actual_losses[1]), np.mean(accs[1])

        eval_loss_1, eval_loss_2, eval_acc_mean = eval_mean(model_1, model_2, loader_test, criterion, args.device)
        val_loss_1, val_loss_2, val_acc_mean = eval_mean(model_1, model_2, loader_val, criterion, args.device)
        if scheduler_1 is not None and scheduler_2 is not None:
            scheduler_1.step()
            scheduler_2.step()
        '''save the models'''
        if best_acc_CCC < eval_acc_mean:
            best_acc_CCC = eval_acc_mean

        print('\n|=======================> EPOCH {:2d}/{:2d} =======================|\n'
              '|-----Loss--Model 1----->>> Meta: {:.4f} | Actual : {:.4f} | Test: {:.4f}\n'
              '|-----Loss--Model 2----->>> Meta: {:.4f} | Actual : {:.4f} | Test: {:.4f}\n'
              '|-----Train ACC------>>> Model_1: {:.2f} | Model_2: {:.2f} |\n'
              '|-----Valid ACC------>>> {:.2f}\n'
              '|-----Test ACC------->>> {:.2f}\n'
              '|-----Best ACC------->>> {:.2f}\n'.format(ep+1, args.epoch_ccc, meta_loss_1, actual_loss_1, eval_loss_1,\
                                                        meta_loss_2, actual_loss_2, eval_loss_2, train_acc_1, train_acc_2,\
                                                        val_acc_mean, eval_acc_mean, best_acc_CCC))
        '''check'''
        # print("CM_range of model 1: {:.4f} to {:.4f}".format(torch.min(model_1.get_mw_params().detach()), torch.max(model_1.get_mw_params().detach())))
        # print("CM_CCC_range of model 1: {} to {:.4f}".format(torch.min(CM_CCC_1), torch.max(CM_CCC_1)))
        # print("CM_range of model 2: {:.4f} to {:.4f}".format(torch.min(model_2.get_mw_params().detach()), torch.max(model_2.get_mw_params().detach())))
        # print("CM_CCC_range of model 1: {:.4f} to {:.4f}".format(torch.min(CM_CCC_2), torch.max(CM_CCC_2)))
        print('|==================================================================|\n\n')

    if args.epoch_ft > 0:
        eval_loss_1, eval_loss_2, eval_acc_mean = eval_mean(model_1, model_2, loader_test, criterion, args.device)
        print('State Before Fine-Tuning:\n'
        'Test Loss   : {:.4f} | {:.4f}\n'
        'Test Acc    : {:.2f}'.format(eval_loss_1, eval_loss_2, eval_acc_mean))

        for e_ft in range(args.epoch_ft):
            '''fine-tune each other'''
            print("Fine-tuning model_1...")
            finetune(args, model_1, meta_loader_2)
            print("Fine-tuning model_2...")
            finetune(args, model_2, meta_loader_1)
        
        eval_loss_1, eval_loss_2, eval_acc_mean = eval_mean(model_1, model_2, loader_test, criterion, args.device)
        print('State After Fine-Tuning:\n'
        'Test Loss   : {:.4f} | {:.4f}\n'
        'Test Acc    : {:.2f}'.format(eval_loss_1, eval_loss_2, eval_acc_mean))
        if best_acc_CCC < eval_acc_mean:
            best_acc_CCC = eval_acc_mean
    


    print("Training with Coupled Confusion Correction Ended at ", time.asctime(time.localtime(time.time())))
    print("Best Test Accuracy during Training: {:.2f}".format(best_acc_CCC))
    print("Last Test Accuracy during Training: {:.2f}".format(eval_acc_mean))
    return best_acc_CCC





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta Pseudo Annotators Framework')

    parser.add_argument('--results_dir', type=str, help='the dir of results', default='./results')
    parser.add_argument('--data_path', type=str, help='the dir of data', default='./data')
    parser.add_argument('--sideinfo_path', type=str, help='the dir of sideinfo of cifar10n', default='./cifar10n_sideinfo')
    parser.add_argument('--dataset', default='cifar10n', type=str, help='dataset')
    parser.add_argument('--no_warm', default=False, action='store_true', help='Whether to warmup the model with Crowd_layer')
    parser.add_argument('--epoch_warmup', default=10, type=int)
    parser.add_argument('--epoch_ccc', default=50, type=int)
    parser.add_argument('--epoch_ft', default=-1, type=int)
    parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--lr_init', default=1e-2, type=float)
    parser.add_argument('--lr_ft', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--arch', default='resnet34', type=str, help='the architecture of the backbone model')
    parser.add_argument('--crct_rate', default=0.5, type=float, help='correction rate')
    parser.add_argument('--bs', default=128, type=int, help='batch size of silver data')
    parser.add_argument('--test_bs', default=32, type=int, help='batch size of test data')
    parser.add_argument('--meta_bs', type=int, default=128, help='batch size of gold data')
    parser.add_argument('--num_workers', default=5, type=int, help='the number of workers in dataloader')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='the ratio to split the training set into training set and validation set')
    parser.add_argument('--meta_size', default=1000, type=int, help='the size of the meta set')
    parser.add_argument('--num_act', default=2, type=int, help='the number of active layers in calculating the meta gradients')
    parser.add_argument('--nb_group', default=30, type=int, help='number of annotator groups')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if not os.path.exists(os.path.join(args.results_dir, args.dataset)):
        os.makedirs(os.path.join(args.results_dir, args.dataset), exist_ok=True)

    print_args(args)
    _ = main(args)



    
