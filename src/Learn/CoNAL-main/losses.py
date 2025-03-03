import torch


def multi_loss(y_true, y_pred, loss_fn=torch.nn.CrossEntropyLoss(reduce='mean').cuda()):
    mask = y_true != -1
    y_pred = torch.transpose(y_pred, 1, 2)
    loss = loss_fn(y_pred[mask], y_true[mask])
    return loss