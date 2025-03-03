import torch
import torch.nn as nn



from model.CL_model import CL_model

from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from fast_pytorch_kmeans import KMeans
from torch.autograd import Variable
from cifar10n import CIFAR10N






def get_model(arch, nclass, in_dim, nb_annotator):
    model_1 = CL_model(arch, nclass, in_dim, nb_annotator)
    model_2 = CL_model(arch, nclass, in_dim, nb_annotator)
    return model_1, model_2


def get_dataset(args):
    train_transform = get_transforms(args, train=True)
    test_transform = get_transforms(args, train=False)

    dst_train = CIFAR10N(data_path=args.data_path, mode='train', transform=train_transform, sideinfo_path=args.sideinfo_path, split_ratio=args.split_ratio)
    dst_test = CIFAR10N(data_path=args.data_path, mode='test', transform=test_transform)
    dst_val = CIFAR10N(data_path=args.data_path, mode='val', transform=test_transform, sideinfo_path=args.sideinfo_path, split_ratio=args.split_ratio)
    nclass = len(np.unique(dst_train.target_gt))
    in_dim = dst_train.data.shape[-1]
    nb_annotator = dst_train.target_cs.shape[1]

    return dst_train, dst_test, dst_val, nclass, in_dim, nb_annotator


def get_optimizer_and_scheduler(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.params(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.1, verbose=True)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.params(), lr=args.lr_init)
        scheduler = None
    else:
        raise NotImplementedError("optimizer type {} is not valid!".format(args.optim))
        
    return optimizer, scheduler



def get_transforms(args, train=True):
    data = args.dataset
    
    if data == "cifar10n":
        TRAIN_MEAN = [0.4914, 0.4822, 0.4465]
        TRAIN_STD = [0.2023, 0.1994, 0.2010]
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        
        if train:
            transform = train_transform
        else:
            transform = test_transform
    else:
        transform = None
    
    return transform

def set_seed(seed=0):
    print(f"Using seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True




def eval(model_1, model_2, loader, criterion, device):
    with torch.no_grad():
        model_1.eval()
        model_2.eval()
        losses_1, correct_1 = 0, 0
        losses_2, correct_2 = 0, 0

        for img, target, _ in loader:
            img, target = img.to(device), target.to(device)
            output_1 = model_1(img)
            output_2 = model_2(img)
            loss_1 = criterion(output_1, target)
            loss_2 = criterion(output_2, target)
            losses_1 += loss_1.item()
            losses_2 += loss_2.item()

            preds_1 = output_1.argmax(1)
            preds_2 = output_2.argmax(1)

            correct_1 += torch.eq(preds_1, target).float().sum().cpu().numpy()
            correct_2 += torch.eq(preds_2, target).float().sum().cpu().numpy()

        acc_1 = correct_1 / len(loader.dataset) * 100
        acc_2 = correct_2 / len(loader.dataset) * 100

        losses_1 /= len(loader)
        losses_2 /= len(loader)
    return losses_1, losses_2, acc_1, acc_2

def eval_mean(model_1, model_2, loader, criterion, device):
    with torch.no_grad():
        model_1.eval()
        model_2.eval()
        losses_1, losses_2 = 0, 0
        correct_mean = 0
        for img, target, _ in loader:
            img, target = img.to(device), target.to(device)
            output_1 = model_1(img)
            output_2 = model_2(img)
            loss_1 = criterion(output_1, target)
            loss_2 = criterion(output_2, target)
            losses_1 += loss_1.item()
            losses_2 += loss_2.item()
            preds_mean = (F.softmax(output_1, dim=-1) + F.softmax(output_2, dim=-1)).argmax(1)

            correct_mean += torch.eq(preds_mean, target).float().sum().cpu().numpy()


        acc_mean = correct_mean / len(loader.dataset) * 100
        losses_1 /= len(loader)
        losses_2 /= len(loader)

    return losses_1, losses_2, acc_mean


def train_CL_1iter(model, img, target_cs_1hot, target_gt, optimizer):
    output, logits = model.forward_CL(img)
    loss = torch.sum(-output * target_cs_1hot) / torch.sum(target_cs_1hot)
    pred = logits.argmax(1)
    acc = torch.eq(pred, target_gt).float().mean().cpu().numpy() * 100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

class get_perclass_sets():
    '''get the img and label per class'''
    def __init__(self, args, train_set) -> None:
        self.train_set = train_set
        self.train_transform = get_transforms(args, train=True)
        self.args = args


    def get_class_sets(self, cls):
        args = self.args
        cls_idx = np.where(self.train_set.target_cs == cls)[0]
        from cifar10n import CIFAR10N_PC as DATASET_PC

        pc_sets = DATASET_PC(data_idx=cls_idx, cls=cls, data_path=args.data_path, mode='train', transform=self.train_transform, sideinfo_path=args.sideinfo_path, split_ratio=args.split_ratio)
        return pc_sets
            






def get_metaset(args, train_set, model_1, model_2, pc_sets):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    nb_sample_perclass = int(args.meta_size / args.nclass)
    meta_idx_1, meta_label_1 = [], []
    meta_idx_2, meta_label_2 = [], []

    print('Distilling meta set')
    '''get the list of max logits and the corresponding label'''
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        for cls in range(args.nclass):
            cls_losses = [[], []]
            '''
            batch including: data, target_cs, target_cs_1hot, target_warmup, target_gt, index
            '''
            cls_set = pc_sets.get_class_sets(cls)
            cls_loader = DataLoader(cls_set, shuffle=False, num_workers=args.num_workers, batch_size=args.bs)
            for _, cls_batch in enumerate(cls_loader):
                img, target = cls_batch[0].to(args.device), cls_batch[1].to(args.device)

                logits_1 = model_1(img)
                logits_2 = model_2(img)
                cls_losses[0].extend(criterion(logits_1, target))
                cls_losses[1].extend(criterion(logits_2, target))
            sel_idx_1 = torch.argsort(torch.tensor(cls_losses[0]))[:nb_sample_perclass]
            sel_idx_2 = torch.argsort(torch.tensor(cls_losses[1]))[:nb_sample_perclass]

            meta_idx_1.extend(cls_loader.dataset.data_idx[sel_idx_1].tolist())
            meta_idx_2.extend(cls_loader.dataset.data_idx[sel_idx_2].tolist())
            meta_label_1.extend(cls_loader.dataset.targets[sel_idx_1].tolist())
            meta_label_2.extend(cls_loader.dataset.targets[sel_idx_2].tolist())


    metaset_acc_1 = (meta_label_1 == train_set.target_gt[meta_idx_1]).sum() / len(meta_idx_1)
    metaset_acc_2 = (meta_label_2 == train_set.target_gt[meta_idx_2]).sum() / len(meta_idx_2)

    print("Dataset |{}| --- Meta data 1 has a size of |{:5d}| with ACC |{:.2f}|".format(args.dataset, len(meta_idx_1),
                                                                                      metaset_acc_1 * 100))
    print("Dataset |{}| --- Meta data 2 has a size of |{:5d}| with ACC |{:.2f}|".format(args.dataset, len(meta_idx_1),
                                                                                      metaset_acc_2 * 100))
    train_transform = get_transforms(args, train=True)

    from cifar10n import CIFAR10NMETA as METASET

    metaset_1 = METASET(meta_idx=meta_idx_1, meta_label=meta_label_1, data_path=args.data_path,  mode='train', transform=train_transform, sideinfo_path=args.sideinfo_path, split_ratio=args.split_ratio)
    metaset_2 = METASET(meta_idx=meta_idx_2, meta_label=meta_label_2, data_path=args.data_path,  mode='train', transform=train_transform, sideinfo_path=args.sideinfo_path, split_ratio=args.split_ratio)
    return metaset_1, metaset_2


'''helper functions (losses, utils etc)'''
def inner_loss(logits, meta_net, CM_CL, target_cs_1hot, w_grad=True):
    '''Annotator Specific Confusion Matrices (ASCMs)'''
    if w_grad == False:
        with torch.no_grad():
            CM_CCC = meta_net(CM_CL)
    else:
        CM_CCC = meta_net(CM_CL)
    batch_prob = F.softmax(logits, dim=-1)
    batch_prob_split = torch.einsum('ik,jkl->ijl', (batch_prob, torch.softmax(CM_CCC, dim=-1)))
    batch_prob_split = torch.log(batch_prob_split + 1e-5)
    loss = torch.sum(-batch_prob_split * target_cs_1hot) / torch.sum(target_cs_1hot)

    return loss

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def _compute_new_grad(args, grads):
    new_grads = []
    for grad in grads[0:-args.num_act]:
        new_grads.append(grad.detach())
    for grad in grads[-args.num_act::]:
        new_grads.append(grad)

    return new_grads

'''training workflow'''

def meta_step(args, model_1, optimizer_1, train_batch, meta_batch, group_label):
    model_1.train()
    '''get the initial Confusion-Correcting-Matrices which is a zero tensor of shape [nb_annotator, nb_classes, nb_classes]'''
    CM_CCC_base = torch.zeros(size=(args.nb_group, args.nclass, args.nclass)).to(args.device)
    CM_CCC_base = Variable(CM_CCC_base, requires_grad=True)
    CM_CCC = CM_CCC_base[group_label]

    data = train_batch[0].to(args.device)
    target_cs_1hot = train_batch[2].to(args.device)

    '''build pseudo model'''
    pseudo_model = CL_model(args.arch, args.nclass, args.in_dim, args.nb_annotator).to(args.device)
    pseudo_model.load_state_dict(model_1.state_dict())
    '''virtual forward'''
    output_pseudo, _ = pseudo_model.forward_CCC(data, CM_CCC)

    loss_pseudo = torch.sum(-output_pseudo * target_cs_1hot) / torch.sum(target_cs_1hot)


    '''virtual backward & update'''
    pseudo_model.zero_grad()
    grads = torch.autograd.grad(loss_pseudo, (pseudo_model.params()), create_graph=True)
    pseudo_lr = optimizer_1.param_groups[0]['lr']
    if args.dataset != 'labelme':
        new_grads = _compute_new_grad(args, grads)
        pseudo_model.update_params(lr_inner=pseudo_lr, source_params=new_grads)
    else:
        pseudo_model.update_params(lr_inner=pseudo_lr, source_params=grads)

    input_meta, target_meta = meta_batch[0], meta_batch[1]
    input_meta, target_meta = input_meta.to(args.device), target_meta.to(args.device)


    '''meta forward'''
    valid_y_f_hat = pseudo_model(input_meta)
    valid_loss = F.cross_entropy(valid_y_f_hat, target_meta)

    grad_CM_CCC = torch.autograd.grad(valid_loss, CM_CCC_base, only_inputs=True)[0]
    CM = pseudo_model.get_mw_params().detach()
    CM_CCC_base = CM_CCC_base - (torch.max(CM) / torch.max(torch.abs(grad_CM_CCC.detach()))) * args.crct_rate * grad_CM_CCC
    del grads
    CM_CCC = CM_CCC_base[group_label]
    return valid_loss.item(), CM_CCC.detach()


def actual_step(args, model_1, optimizer_1, train_batch, CM_CCC):
    # actual forward & backward & update
    model_1.train()
    data, target_cs_1hot, target_gt = train_batch[0].to(args.device), train_batch[2].to(args.device), train_batch[3].to(args.device)
    output_actual, logits_actual = model_1.forward_CCC(data, CM_CCC)
    loss_actual = torch.sum(-output_actual * target_cs_1hot) / torch.sum(target_cs_1hot)
    pred = logits_actual.argmax(1)
    acc = torch.eq(pred, target_gt).float().mean().cpu().numpy() * 100
    optimizer_1.zero_grad()
    loss_actual.backward()
    optimizer_1.step()
    return loss_actual.item(), acc

def finetune(args, model, meta_loader):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.params(), lr=args.lr_ft)
    model.train()
    for ep_ft in range(args.epoch_ft):
        losses_ft, accs_ft = [], []
        for i, meta_batch in enumerate(meta_loader):
            input_meta, target_meta = meta_batch[0], meta_batch[1]
            input_meta, target_meta = input_meta.to(args.device), target_meta.to(args.device)
            logits_meta = model(input_meta)
            loss = criterion(logits_meta, target_meta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = logits_meta.argmax(1)
            acc = torch.eq(pred, target_meta).float().mean().cpu().numpy() * 100
            losses_ft.append(loss.item())
            accs_ft.append(acc)
        print("Fine-Tune Epoch |{:2d}/{:2d}| Loss: |{:.4f}| ACC: |{:.2f}|".format(ep_ft+1, args.epoch_ft,\
                                        np.mean(losses_ft), np.mean(accs_ft)))
        

def print_args(args):
    for key, value in zip(args.__dict__.keys(), args.__dict__.values()):
        print("{:14s} : {}".format(key, value))

def set_seed(seed=0):
    print(f"Using seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True


