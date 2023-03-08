import os
import math
import time
import random
import shutil
import argparse
import builtins
import warnings
import numpy as np

import torch
import torch.nn 
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms 

from utils.utils_loss import *
from utils.utils_algo import *
from utils.models import linear, mlp
from cifar_models import densenet, convnet
from cifar_models.resnet import CIFAR_ResNet
from datasets.kmnist import load_kmnist
from datasets.cifar10 import load_cifar10
from datasets.cifar100 import load_cifar100

def train(args, epoch, train_loader, net, loss_fn, optimizer):

    total_num = 0
    bingo_num = 0
    total_indexes = []
    total_plabels = []
    total_dlabels = []
    total_classfy_out = []
    total_classfy_logit = []

    net.train()
    ###### ---------------- one epoch training ---------------- ######
    for i, (images, plabels, dlabels, index) in enumerate(train_loader): 

        images  = images.cuda()
        plabels = plabels.float().cuda()
        dlabels = dlabels.long().detach().cuda() # only for evalaution
        index   = index.cuda()

        # train and save results\
        outputs = net(images)[0]
        classfy_out = F.softmax(outputs, dim=1)
        total_num += plabels.size(0)
        bingo_num  += torch.eq(torch.max(classfy_out, 1)[1], dlabels).sum().cpu()
        total_indexes.append(index.detach().cpu().numpy())
        total_plabels.append(plabels.detach().cpu().numpy())
        total_dlabels.append(dlabels.detach().cpu().numpy())
        total_classfy_out.append(classfy_out.detach().cpu().numpy())
        total_classfy_logit.append(outputs.detach().cpu().numpy())

        # calculate loss and update model
        if args.loss_type in ['rc', 'proden']:
            average_loss = loss_fn(args, outputs, index)
        elif args.loss_type == 'cc':
            average_loss = loss_fn(args, outputs, plabels)
        elif args.loss_type in ['lws', 'lwc']:
            average_loss = loss_fn(args, outputs, plabels, index)
        elif args.loss_type in ['log']:
            average_loss = loss_fn(args, outputs, plabels)

        optimizer.zero_grad()
        average_loss.backward()
        optimizer.step()

        # update args.confidence
        if args.loss_type in ['rc', 'proden']:
            with torch.no_grad():
                args.confidence[index, :] = classfy_out * plabels
                base_value = args.confidence.sum(dim=1).unsqueeze(1).repeat(1, args.confidence.shape[1])
                args.confidence = args.confidence / base_value
        elif args.loss_type in ['lws', 'lwc']:
            with torch.no_grad():
                onezero = torch.zeros(classfy_out.shape[0], classfy_out.shape[1])
                onezero[plabels > 0] = 1
                counter_onezero = 1 - onezero
                onezero = onezero.cuda()
                counter_onezero = counter_onezero.cuda()
                new_weight1 = classfy_out * onezero
                new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(args.confidence.shape[1], 1).transpose(0, 1)
                new_weight2 = classfy_out * counter_onezero
                new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(args.confidence.shape[1], 1).transpose(0, 1)
                new_weight = new_weight1 + new_weight2
                args.confidence[index, :] = new_weight

    ###### ---------------- end of one epoch training ---------------- ######
    temp_dlabels = train_loader.dataset.dlabels.astype('int')
    temp_plabels = train_loader.dataset.plabels
    num_sample = len(temp_dlabels)
    epoch_partial_rate = np.mean(np.sum(temp_plabels, axis=1))
    epoch_bingo_rate = np.sum(temp_plabels[np.arange(num_sample), temp_dlabels] == 1.0)/num_sample

    epoch_train_acc = bingo_num/total_num
    total_indexes = np.concatenate(total_indexes)
    total_plabels = np.concatenate(total_plabels)
    total_dlabels = np.concatenate(total_dlabels)
    total_classfy_out = np.concatenate(total_classfy_out)
    total_classfy_logit = np.concatenate(total_classfy_logit)
    print (f'Epoch:{epoch}/{args.epochs} Train classification acc={epoch_train_acc:.4f} Partial: {epoch_partial_rate:.4f} Bingo: {epoch_bingo_rate:.4f}')

    train_save = {
        'epoch_train_acc':      epoch_train_acc,
        'epoch_bingo_rate':     epoch_bingo_rate,
        'epoch_partial_rate':   epoch_partial_rate,
        'total_indexes':        total_indexes,
        'total_plabels':        total_plabels,
        'total_dlabels':        total_dlabels,
        'total_classfy_out':    total_classfy_out,
        'total_classfy_logit':  total_classfy_logit,
    }
    return train_save


def test(args, epoch, test_loader, net):
    
    bingo_num = 0
    total_num = 0
    test_probs = []
    test_preds = []
    test_labels = []
    test_hidden = []
    
    net.eval()
    for images, _, dlabels, _ in test_loader:
        images = images.cuda()
        dlabels = dlabels.cuda()
        outputs, hiddens = net(images)
        outputs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs.data, 1) 
        total_num += images.size(0)
        bingo_num += (pred == dlabels).sum().item()
        test_preds.append(pred.cpu().numpy())
        test_probs.append(outputs.detach().cpu().numpy())
        test_labels.append(dlabels.cpu().numpy())
        test_hidden.append(hiddens.detach().cpu().numpy())

    epoch_test_acc = bingo_num / total_num
    print(f'Epoch={epoch}/{args.epochs} Test accuracy={epoch_test_acc:.4f}, bingo_num={bingo_num},  total_num={total_num}')
    test_probs = np.concatenate(test_probs)
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_hidden = np.concatenate(test_hidden)
    test_save = {
        'test_probs':       test_probs,
        'test_preds':       test_preds,
        'test_labels':      test_labels,
        'test_hidden':      test_hidden,
        'epoch_test_acc':   epoch_test_acc,
    }

    return epoch_test_acc, test_save


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='PLL Baseline Model')
    
    ## input parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name (cifar10)')
    parser.add_argument('--partial_rate', default=0.0, type=float, help='ambiguity level (q)')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='noise level (gt may not in partial set)')
    parser.add_argument('--workers', default=10, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')

    ## model parameters
    parser.add_argument('--encoder', default='resnet', type=str, help='encoder: resnet, mlp, ...')
    parser.add_argument('--low_dim', default=128, type=int, help='embedding dimension for resnet')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate for convnet')
    parser.add_argument('--num_class', default=10, type=int, help='number of classes in the dataset.')
    parser.add_argument('--loss_type', help='specify a loss function', default='rc', type=str)
    parser.add_argument('--lws_weight1', help='weight for first  item in [lws, lwc]', default=1, type=float)
    parser.add_argument('--lws_weight2', help='weight for second item in [lws, lwc]', default=1, type=float)

    ## training parameters
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_adjust', default='Case1', type=str, help='Learning rate adjust manner: Case1 or Case2.')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay (default: 1e-5).')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--decaystep', help='learning rate\'s decay step', type=int, default=10) # adjust learning rate
    parser.add_argument('--decayrate', help='learning rate\'s decay rate', type=float, default=0.9)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether to save whole results')
    parser.add_argument('--save_root', help='where to save results', default='./savemodels', type=str)
    args = parser.parse_args()
    print(args)

    cudnn.benchmark = True
    torch.set_printoptions(precision=2, sci_mode=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.gpu)

    print (f'====== Step1: Reading Data =======')
    train_loader, train_givenY, test_loader = [], [], []
    if args.dataset == 'cifar10':
        input_channels = 3
        args.num_class = 10
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])
        train_loader, train_givenY, test_loader = load_cifar10(args, transform)
    elif args.dataset == 'cifar100':
        input_channels = 3
        args.num_class = 100
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),])
        train_loader, train_givenY, test_loader = load_cifar100(args, transform)
    elif args.dataset == 'kmnist':
        num_features = 28 * 28
        args.num_class = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1918), (0.3483))])
        train_loader, train_givenY, test_loader = load_kmnist(args, transform)
    print (f'training samples: {len(train_loader.dataset)}')
    print (f'testing samples: {len(test_loader.dataset)}')

    print (f'====== Step2: Gaining model and optimizer =======')
    # loss function and args.confidence
    train_givenY = torch.FloatTensor(train_givenY)
    if args.loss_type in ['rc', 'proden']:
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        args.confidence = train_givenY.float() / tempY
        args.confidence = args.confidence.cuda()
        loss_fn = rc_loss
    elif args.loss_type == 'cc':
        loss_fn = cc_loss
    elif args.loss_type == 'lws':
        n, c = train_givenY.shape[0], train_givenY.shape[1]
        args.confidence = torch.ones(n, c) / c # generate args.confidence with all ones
        args.confidence = args.confidence.cuda()
        loss_fn = lws_loss
    elif args.loss_type == 'lwc':
        n, c = train_givenY.shape[0], train_givenY.shape[1]
        args.confidence = torch.ones(n, c) / c # generate args.confidence with all ones
        args.confidence = args.confidence.cuda()
        loss_fn = lwc_loss
    elif args.loss_type == 'log':
        loss_fn = log_loss

    # encoder
    if args.encoder == 'linear':
        net = linear(n_inputs=num_features, n_outputs=args.num_class)
    elif args.encoder == 'mlp':
        net = mlp(n_inputs=num_features, n_outputs=args.num_class)
    elif args.encoder == 'convnet':
        net = convnet(input_channels=input_channels, n_outputs=args.num_class, dropout_rate=args.dropout_rate)
    elif args.encoder == 'resnet':
        net = CIFAR_ResNet(feat_dim=args.low_dim, num_class=args.num_class)
    elif args.encoder == 'densenet':
        net = densenet(num_classes=args.num_class)
    net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, 
                                momentum=0.9, 
                                weight_decay=args.weight_decay, )

    print (f'====== Step3: Training and Evaluation =======')
    test_accs = []
    all_labels = []
    for epoch in range(1, args.epochs+1):
        if args.lr_adjust == 'case1':
            adjust_learning_rate_V1(args, optimizer, epoch)
        elif args.lr_adjust == 'case2':
            adjust_learning_rate_V2(args, optimizer, epoch)
        train_save = train(args, epoch, train_loader, net, loss_fn, optimizer)
        test_acc, test_save = test(args, epoch, test_loader, net)
        test_accs.append(test_acc)

        # save results
        all_labels.append({'epoch_train_acc':    train_save['epoch_train_acc'],
                           'epoch_bingo_rate':   train_save['epoch_bingo_rate'],
                           'epoch_partial_rate': train_save['epoch_partial_rate'],
                           'epoch_test_acc' :    test_save['epoch_test_acc'],
                           })
        if args.savewhole and epoch%20==0: # further save data which occupy much space
            all_labels[-1]['total_plabels'] = train_save['total_plabels']
            all_labels[-1]['total_dlabels'] = train_save['total_dlabels']
            all_labels[-1]['total_classfy_logit'] = train_save['total_classfy_logit']


    print (f'====== Step4: Saving =======')
    if args.loss_type in ['rc', 'proden', 'lws', 'lwc']:
        args.confidence = args.confidence.detach().cpu().numpy()
    save_root = args.save_root
    if not os.path.exists(save_root): os.makedirs(save_root)

    ## gain suffix_name
    suffix_name = f'{args.dataset}_modelname:{args.loss_type}_plrate:{args.partial_rate}_noiserate:{args.noise_rate}_model:{args.encoder}'
    ## gain res_name
    best_index = np.argmax(np.array(test_accs))
    bestacc = test_accs[best_index]
    res_name = f'testacc:{bestacc}'

    save_path = f'{save_root}/{suffix_name}_{res_name}_{time.time()}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        all_labels=all_labels)
