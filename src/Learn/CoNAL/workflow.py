from losses import *
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import IPython

loss_fn = torch.nn.CrossEntropyLoss(reduce='mean').cuda()
def train(train_loader, model, optimizer, criterion=F.cross_entropy, mode='simple', annotators=None, pretrain=None,
          support = None, support_t = None, scale=0):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    loss = 0

    correct_rec = 0
    total_rec = 0
    for idx, input, targets, targets_onehot, true_labels in train_loader:
        input = input.cuda()
        targets = targets.cuda().long()
        targets_onehot = targets_onehot.cuda()
        targets_onehot[targets_onehot == -1] = 0
        true_labels = true_labels.cuda().long()

        if mode == 'simple':
            loss = 0
            if scale:
                cls_out, output, trace_norm = model(input)
                loss += scale * trace_norm
                mask = targets != -1
                y_pred = torch.transpose(output, 1, 2)
                y_true = torch.transpose(targets_onehot, 1, 2).float()
                loss += torch.mean(-y_true[mask] * torch.log(y_pred[mask]))
            else:
                cls_out, output = model(input)
                loss += criterion(targets, output)
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
        elif mode == 'common':
            rec_loss = 0
            loss = 0
            cls_out, output = model(input, mode='train')
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
            loss += criterion(targets, output)
            loss -= 0.00001 * torch.sum(torch.norm((model.kernel - model.common_kernel).view(targets.shape[1], -1), dim=1, p=2))
        else:
            output, _ = model(input)
            loss = loss_fn(output, true_labels)
            _, predicted = output.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if mode =='simple' or mode == 'common':
        print('Training acc: ', correct / total)
        return correct / total


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    target = []
    predict = []
    for _, inputs, targets in test_loader:
        inputs = inputs.cuda()
        target.extend(targets.data.numpy())
        targets = targets.cuda()

        total += targets.size(0)
        output, _ = model(inputs, mode='test')
        _, predicted = output.max(1)
        predict.extend(predicted.cpu().data.numpy())
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    f1 = f1_score(target, predict, average='macro')

    classes = list(set(target))
    classes.sort()
    acc_per_class = []
    predict = np.array(predict)
    target = np.array(target)
    for i in range(len(classes)):
        instance_class = target == i
        acc_i = np.mean(predict[instance_class] == classes[i])
        acc_per_class.append(acc_i)
    return acc, f1, acc_per_class




