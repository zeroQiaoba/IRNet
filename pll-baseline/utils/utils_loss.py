import torch 
import torch.nn.functional as F
import numpy as np
import math


def rc_loss(args, outputs, index, ood_weight=None):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * args.confidence[index, :]
    if ood_weight is None:
        average_loss = -((final_outputs).sum(dim=1)).mean()
    else:
        average_loss = (-((final_outputs).sum(dim=1)) * ood_weight).mean()
    return average_loss


def cc_loss(args, outputs, partialY, ood_weight=None):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    if ood_weight is None:
        average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    else:
        average_loss = (-torch.log(final_outputs.sum(dim=1)) * ood_weight).mean()
    return average_loss


def lws_loss(args, outputs, partialY, index, ood_weight=None):
    device = outputs.device

    # (onezero, counter_onezero)
    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1     # partial mask
    counter_onezero = 1 - onezero # nonpartial mask
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    # onezero -> partial part loss (use sigmoid to calculate loss function)
    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (1 + torch.exp(-outputs[outputs > 0]))
    l1 = args.confidence[index, :] * onezero * sig_loss1
    if ood_weight is None:
        average_loss1 = torch.sum(l1) / l1.size(0)
    else:
        average_loss1 = (torch.sum(l1, axis=1) * ood_weight).mean()

    # counter_onezero -> non-partial part loss
    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = args.confidence[index, :] * counter_onezero * sig_loss2
    if ood_weight is None:
        average_loss2 = torch.sum(l2) / l2.size(0)
    else:
        average_loss2 = (torch.sum(l2, axis=1) * ood_weight).mean()

    # weighted loss
    average_loss = args.lws_weight2 * average_loss1 + args.lws_weight1 * average_loss2
    return average_loss


def lwc_loss(args, outputs, partialY, index, ood_weight=None):
    device = outputs.device

    ## (onezero, counter_onezero)
    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sm_outputs = F.softmax(outputs, dim=1)

    ## for partial loss (min) + weight (use ce to calculate loss function)
    sig_loss1 = - torch.log(sm_outputs + 1e-8)
    l1 = args.confidence[index, :] * onezero * sig_loss1
    if ood_weight is None:
        average_loss1 = torch.sum(l1) / l1.size(0)
    else:
        average_loss1 = (torch.sum(l1, axis=1) * ood_weight).mean()

    ## for non-partial loss (max) + weight
    sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
    l2 = args.confidence[index, :] * counter_onezero * sig_loss2
    if ood_weight is None:
        average_loss2 = torch.sum(l2) / l2.size(0)
    else:
        average_loss2 = (torch.sum(l2, axis=1) * ood_weight).mean()

    average_loss = args.lws_weight2 * average_loss1 + args.lws_weight1 * average_loss2
    return average_loss


def log_loss(args, outputs, partialY, ood_weight=None):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float() # n
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    # average_loss = - ((k-1)/(k-can_num) * torch.log(final_outputs.sum(dim=1))).mean() # for random partial label, it will lead to nan
    if ood_weight is None:
        average_loss = - (torch.log(final_outputs.sum(dim=1))).mean()
    else:
        average_loss = (- (torch.log(final_outputs.sum(dim=1)))*ood_weight).mean()
    return average_loss

def exp_loss(args, outputs, partialY, ood_weight=None):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float() # n
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    # average_loss = ((k-1)/(k-can_num) * torch.exp(-final_outputs.sum(dim=1))).mean() # for random partial label, it will cause nan
    if ood_weight is None:
        average_loss = (torch.exp(-final_outputs.sum(dim=1))).mean()
    else:
        average_loss = ((torch.exp(-final_outputs.sum(dim=1)))*ood_weight).mean()
    return average_loss




# Y is onehot-version GT
def ce_loss(outputs, Y):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss
    
## 直接让输出sm_outputs与partialY计算距离
def mae_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = torch.nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss

## 直接让输出sm_outputs与partialY计算距离
def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss

def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q # n
    return sample_loss


# outputs: [batch_size, num_class]
# Y: onehot [1, 0, 0, 0, ..., 0]
def phuber_ce_loss(outputs, Y):
    device = outputs.device

    trunc_point = 10
    n = Y.shape[0] # n=batch size
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * Y
    final_confidence = final_outputs.sum(dim=1) # p(y|x)
    
    # save results in sample_loss (each sample has one loss value)
    sample_loss = torch.zeros(n).to(device)

    ce_index = (final_confidence > 1/trunc_point) # sample index
    if ce_index.sum() > 0: # has sample satisfy this condition
        ce_outputs = outputs[ce_index, :]
        logsm_outputs =  F.log_softmax(ce_outputs, dim=1)
        final_ce_outputs = logsm_outputs * Y[ce_index,:]
        sample_loss[ce_index] = - final_ce_outputs.sum(dim=-1)

    linear_index = (final_confidence <= 1/trunc_point)
    if linear_index.sum() > 0: # has sample satisfy this condition
        sample_loss[linear_index] = math.log(trunc_point) - trunc_point*final_confidence[linear_index] + 1

    return sample_loss


# 为了计算partial-level loss, 我们需要假设某一个class是GT，然后计算得到损失
def unbiased_estimator(loss_fn, outputs, partialY):
    device = outputs.device
    n, k = partialY.shape[0], partialY.shape[1] # n=batch size; k=class number
    comp_num = k - partialY.sum(dim=1)
    temp_loss = torch.zeros(n, k).to(device)
    for i in range(k):
        tempY = torch.zeros(n, k).to(device)
        tempY[:, i] = 1.0
        temp_loss[:, i] = loss_fn(outputs, tempY) # 假设某一个class是GT，然后计算得到损失

    ## 然后将两部分损失加权
    candidate_loss = (temp_loss * partialY).sum(dim=1)
    noncandidate_loss = (temp_loss * (1-partialY)).sum(dim=1)
    # total_loss = candidate_loss - (k-comp_num-1.0)/comp_num * noncandidate_loss
    total_loss = candidate_loss - noncandidate_loss
    average_loss = total_loss.mean()
    return average_loss

