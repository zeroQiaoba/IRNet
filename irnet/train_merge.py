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

from pico import PiCO
from models.cifar_resnet import CIFAR_ResNet
from models.cifar_preactresnet import CIFAR_PreActResNet
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss
from datasets.cifar10 import load_cifar10
from datasets.cifar100 import load_cifar100
from datasets.kmnist import load_kmnist

def train(args, epoch, train_loader, model, loss_fn, loss_cont_fn, optimizer):

    total_num = 0
    cls_bingo_num = 0
    cons_bingo_num = 0
    total_indexes = []
    total_plabels = []
    total_dlabels = []
    total_classfy_out = []
    total_cluster_out = []

    model.train()
    ## images_w1: [3, 32, 32]
    for i, (images_w1, images_w2, images_w3, images_s1, images_s2, images_s3, plabels, dlabels, index) in enumerate(train_loader):
        
        # print (f'itertion {i}/{len(train_loader)}')
        images_w1 = images_w1.cuda()
        images_w2 = images_w2.cuda()
        images_w3 = images_w3.cuda() 
        images_s1 = images_s1.cuda()
        images_s2 = images_s2.cuda()
        images_s3 = images_s3.cuda()
        plabels = plabels.cuda()
        dlabels = dlabels.long().detach().cuda() # only for evalaution
        index = index.cuda()
        
        # train and save results
        classfy_out, cluster_out, cont_features, cont_labels = model(images_w1, images_s1, plabels, args)
        #if epoch >= args.correct_start: # augmentation is only utilized in the correction process
        if args.augmentation_type in ['case3', 'case5', 'case6']: classfy_augw2, cluster_augw2, _, _ = model(images_w2, eval_only=True) # augmentated predict results
        if args.augmentation_type in ['case6'                  ]: classfy_augw3, cluster_augw3, _, _ = model(images_w3, eval_only=True)
        if args.augmentation_type in ['case2', 'case4', 'case5']: classfy_augs2, cluster_augs2, _, _ = model(images_s2, eval_only=True)
        if args.augmentation_type in ['case4'                  ]: classfy_augs3, cluster_augs3, _, _ = model(images_s3, eval_only=True)
        total_num += plabels.size(0)
        cls_bingo_num  += torch.eq(torch.max(classfy_out, 1)[1], dlabels).sum().cpu()
        cons_bingo_num += torch.eq(torch.max(cluster_out, 1)[1], dlabels).sum().cpu()
        total_indexes.append(index.detach().cpu().numpy())
        total_plabels.append(plabels.detach().cpu().numpy())
        total_dlabels.append(dlabels.detach().cpu().numpy())
        total_classfy_out.append(classfy_out.detach().cpu().numpy())
        total_cluster_out.append(cluster_out.detach().cpu().numpy())

        # loss function
        batch_size = classfy_out.shape[0]
        cont_labels = cont_labels.contiguous().view(-1, 1)
        cont_mask = torch.eq(cont_labels[:batch_size], cont_labels.T).float().cuda() # mask for SupCon
        if epoch >= args.proto_start: # update confidence
            if args.proto_type=='cluster':  pred = cluster_out
            if args.proto_type=='classify': pred = classfy_out
            loss_fn.confidence_update(args, pred, index, plabels)
        loss_cont = loss_cont_fn(features=cont_features, mask=cont_mask, batch_size=batch_size)
        loss_cls = loss_fn(args, classfy_out, index) # need preds
        loss = loss_cls + args.loss_weight * loss_cont

        # whether add mixup process
        if args.mixup_flag:
            input_a, target_a = images_w1, loss_fn.confidence[index, :]
            random_idx = torch.randperm(input_a.size(0))
            input_b, target_b = input_a[random_idx], target_a[random_idx]
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            input_mix = lam * input_a + (1-lam) * input_b
            target_mix = lam * target_a + (1-lam) * target_b
            classfy_mix, cluster_mix, _, _ = model(input_mix, eval_only=True)
            mixup_loss = - ((torch.log(classfy_mix) * target_mix).sum(dim=1)).mean()
            loss = loss + args.mixup_weight * mixup_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.correct_start and epoch <= (args.correct_start + args.correct_duration):

            ## update train_loader.dataset.plabels
            if args.augmentation_type == 'case1':
                if args.correct_type=='cluster':  pred1, pred2, pred3 = cluster_out, cluster_out, cluster_out
                if args.correct_type=='classify': pred1, pred2, pred3 = classfy_out, classfy_out, classfy_out
            elif args.augmentation_type == 'case2':
                if args.correct_type=='cluster':  pred1, pred2, pred3 = cluster_out, cluster_augs2, cluster_augs2
                if args.correct_type=='classify': pred1, pred2, pred3 = classfy_out, classfy_augs2, classfy_augs2
            elif args.augmentation_type == 'case3':
                if args.correct_type=='cluster':  pred1, pred2, pred3 = cluster_out, cluster_augw2, cluster_augw2
                if args.correct_type=='classify': pred1, pred2, pred3 = classfy_out, classfy_augw2, classfy_augw2
            elif args.augmentation_type == 'case4':
                if args.correct_type=='cluster':  pred1, pred2, pred3 = cluster_out, cluster_augs2, cluster_augs3
                if args.correct_type=='classify': pred1, pred2, pred3 = classfy_out, classfy_augs2, classfy_augs3
            elif args.augmentation_type == 'case5':
                if args.correct_type=='cluster':  pred1, pred2, pred3 = cluster_out, cluster_augw2, cluster_augs2
                if args.correct_type=='classify': pred1, pred2, pred3 = classfy_out, classfy_augw2, classfy_augs2
            elif args.augmentation_type == 'case6':
                if args.correct_type=='cluster':  pred1, pred2, pred3 = cluster_out, cluster_augw2, cluster_augw3
                if args.correct_type=='classify': pred1, pred2, pred3 = classfy_out, classfy_augw2, classfy_augw3
            p1_maxval, p1_maxlabel = (pred1 * plabels).max(dim=1)
            nonp1_maxval, nonp1_maxlabel = (pred1 * (1 - plabels)).max(dim=1)
            p2_maxval, p2_maxlabel = (pred2 * plabels).max(dim=1)
            nonp2_maxval, nonp2_maxlabel = (pred2 * (1 - plabels)).max(dim=1)
            p3_maxval, p3_maxlabel = (pred3 * plabels).max(dim=1)
            nonp3_maxval, nonp3_maxlabel = (pred3 * (1 - plabels)).max(dim=1)

            if args.correct_threshold_auto == True: # such process only change the cretria for noisy sample detection
                assert args.correct_type=='classify', f'only support for classify output'
                threshold = auto_correct_threshold(args)
                select_index = torch.logical_and((torch.log(p1_maxval) - torch.log(nonp1_maxval) < math.log(threshold)), (torch.log(p2_maxval) - torch.log(nonp2_maxval) < math.log(threshold)))
                select_index = torch.logical_and(select_index, (torch.log(p3_maxval) - torch.log(nonp3_maxval) < math.log(threshold)))
                select_index = torch.logical_and(select_index, (nonp1_maxlabel == nonp2_maxlabel))
                select_index_update = torch.logical_and(select_index, (nonp1_maxlabel == nonp3_maxlabel))
            else:
                threshold = adjust_correct_threshold(args, epoch)
                select_index = torch.logical_and((nonp1_maxval > p1_maxval + threshold), (nonp2_maxval > p2_maxval + threshold))
                select_index = torch.logical_and(select_index, (nonp3_maxval > p3_maxval + threshold))
                select_index = torch.logical_and(select_index, (nonp1_maxlabel == nonp2_maxlabel))
                select_index_update = torch.logical_and(select_index, (nonp1_maxlabel == nonp3_maxlabel))
            correct_index = index[select_index_update]
            correct_label = nonp1_maxlabel[select_index_update]
            correct_index = correct_index.cpu().numpy()
            correct_label = correct_label.cpu().numpy()
            train_loader.dataset.plabels[correct_index, correct_label] = 1.0
            
            # => whether utilize deletion process
            if args.correct_deletion == True:
                assert args.correct_update == 'case3'
                modify_one = torch.ones(pred1.size()).cuda() * 10000.0
                modify_pred1 = torch.where(plabels==1, pred1, modify_one)
                p1_minval, p1_minlabel = modify_pred1.min(dim=1)
                deletion_label = p1_minlabel[select_index_update]
                deletion_label = deletion_label.cpu().numpy()
                train_loader.dataset.plabels[correct_index, deletion_label] = 0.0

            ## update loss_fn.confidence
            # case1: uniform distribution as initial ones
            if args.correct_update == 'case1':
                plabels = train_loader.dataset.plabels[index.cpu().numpy(), :]
                plabels = torch.FloatTensor(plabels).cuda()
                plabels_sum = plabels.sum(dim=1).unsqueeze(1).repeat(1, plabels.shape[1])
                confidence_temp = plabels.float() / plabels_sum
                confidence_temp = confidence_temp.cuda()
                loss_fn.confidence[correct_index] = confidence_temp[select_index_update].detach()
            # case2: only change new added plabel to 1/args.num_class
            elif args.correct_update == 'case2' and len(correct_index) > 0:
                loss_fn.confidence[correct_index, correct_label] = 1/args.num_class
                confidence_sum = loss_fn.confidence[correct_index, :].sum(dim=1).unsqueeze(1).repeat(1, args.num_class)
                loss_fn.confidence[correct_index, :] /= confidence_sum
            # case3: only change new added plabel to nonp1_maxval
            elif args.correct_update == 'case3' and len(correct_index) > 0:
                loss_fn.confidence[correct_index, correct_label] = nonp1_maxval[select_index_update].float().detach()
                # => whether utilize deletion process
                if args.correct_deletion == True: loss_fn.confidence[correct_index, deletion_label] = 0
                confidence_sum = loss_fn.confidence[correct_index, :].sum(dim=1).unsqueeze(1).repeat(1, args.num_class)
                loss_fn.confidence[correct_index, :] /= confidence_sum

            ## print results
            if i % 300 == 0:
                print (f'=== [With correction] Iter: {i}/{len(train_loader)}   Threshold: {threshold}   modified number: {len(correct_index)} ===')
                np.set_printoptions(precision=3)
                pred = cluster_out.detach().cpu().numpy()[0]
                print ('cluster_out:', pred, 'diff: ', max(pred)-min(pred))
                pred = classfy_out.detach().cpu().numpy()[0]
                print ('classfy_out:', pred, 'diff: ', max(pred)-min(pred))
                temp_dlabels = train_loader.dataset.dlabels.astype('int')
                temp_plabels = train_loader.dataset.plabels
                num_sample = len(temp_dlabels)
                print('Average candidate num: ', np.mean(np.sum(temp_plabels, axis=1))) # initial value = 5.5
                bingo_number = np.sum(temp_plabels[np.arange(num_sample), temp_dlabels] == 1.0)
                print(f'Average bingo rate: {bingo_number/num_sample:.4f}   {bingo_number}/{num_sample}') # initial value = 0.8
        ## end of iter


    ## end of epoch, print and save results
    temp_dlabels = train_loader.dataset.dlabels.astype('int')
    temp_plabels = train_loader.dataset.plabels
    num_sample = len(temp_dlabels)
    epoch_partial_rate = np.mean(np.sum(temp_plabels, axis=1))
    epoch_bingo_rate = np.sum(temp_plabels[np.arange(num_sample), temp_dlabels] == 1.0)/num_sample

    epoch_cls_acc = cls_bingo_num/total_num
    epoch_cont_acc = cons_bingo_num/total_num
    total_indexes = np.concatenate(total_indexes)
    total_plabels = np.concatenate(total_plabels)
    total_dlabels = np.concatenate(total_dlabels)
    total_classfy_out = np.concatenate(total_classfy_out)
    total_cluster_out = np.concatenate(total_cluster_out)

    print (f'Epoch={epoch}/{args.epochs} Train classification acc={epoch_cls_acc:.4f} contrastive acc={epoch_cont_acc:.4f} Partial: {epoch_partial_rate:.4f} Bingo: {epoch_bingo_rate:.4f}')
    train_save = {
        'epoch_partial_rate':   epoch_partial_rate,
        'epoch_bingo_rate':     epoch_bingo_rate,
        'epoch_cls_acc':        epoch_cls_acc,
        'epoch_cont_acc':       epoch_cont_acc,
        'total_indexes':        total_indexes,
        'total_plabels':        total_plabels,
        'total_dlabels':        total_dlabels,
        'total_classfy_out':    total_classfy_out,
        'total_cluster_out':    total_cluster_out,
    }
    return train_save


def test(args, epoch, test_loader, model):
    test_preds = []
    test_labels = []
    test_probs = []
    test_hidden1 = []
    test_hidden2 = []
    with torch.no_grad():     
        model.eval()
        bingo_num = 0
        total_num = 0
        # images: [3, 32, 32]
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            logit, _, hidden1, hidden2 = model(images, eval_only=True)
            _, predicts = torch.max(logit, 1)
            total_num += images.size(0)
            bingo_num += torch.eq(predicts, labels).sum().cpu()
            test_preds.append(predicts.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_probs.append(logit.cpu().numpy())
            test_hidden1.append(hidden1.cpu().numpy())
            test_hidden2.append(hidden2.cpu().numpy())
        test_acc = bingo_num / total_num
        print(f'Epoch={epoch}/{args.epochs} Test accuracy={test_acc:.4f}, bingo_num={bingo_num},  total_num={total_num}')
        test_hidden1 = np.concatenate(test_hidden1)
        test_hidden2 = np.concatenate(test_hidden2)
        test_probs = np.concatenate(test_probs)
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        test_save = {
            'test_hidden1': test_hidden1,
            'test_hidden2': test_hidden2,
            'test_probs': test_probs,
            'test_preds': test_preds,
            'test_labels': test_labels
        }

    return test_acc, test_save
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch implementation of noise partial label learning')

    ## -------------- input parameters --------------
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name (cifar10)')
    parser.add_argument('--partial_rate', default=0.0, type=float, help='ambiguity level (q)')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='noise level (gt in non-candidate set)')
    parser.add_argument('--workers', default=6, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')

    ## -------------- model parameters --------------
    parser.add_argument('--encoder', default='resnet', type=str, help='encoder: preact or resnet')
    parser.add_argument('--low_dim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--num_class', default=10, type=int, help='number of class')
    parser.add_argument('--moco_m', default=0.999, type=float, help='momentum for updating momentum encoder')
    parser.add_argument('--moco_queue', default=8192, type=int, help='queue size; number of negative samples')
    parser.add_argument('--loss_weight', default=0.5, type=float, help='contrastive loss weight')
    parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str, help='pseudo target updating coefficient (phi)')

    # -------------- for confidence update --------------
    parser.add_argument('--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes')
    parser.add_argument('--proto_start', default=1, type=int, help = 'Start Prototype Updating')
    parser.add_argument('--proto_type', default='cluster', type=str, help = 'Correct type: cluster or classify')
    parser.add_argument('--proto_case', default='Case1', type=str, help = 'Correct case: Case1(onehot update) or Case2(prob update)')

    # -------------- for loss function --------------
    parser.add_argument('--loss_type', default='CE', type=str, help='loss type in training: CE, CC, EXP, LWC, MAE, MSE, SCE, GCE')
    parser.add_argument('--lwc_weight', default=1.0, type=float, help='weight in lwc loss, choose from [1,2,3]')
    parser.add_argument('--sce_alpha', default=0.1, type=float, help='alpha in rec loss, choose from [0.01, 0.1, 1, 6]')
    parser.add_argument('--sce_beta', default=1.0, type=float, help='beta in rec loss, choose from [0.1, 1.0]')
    
    # -------------- for mixup --------------
    parser.add_argument('--mixup_flag', action='store_true', default=False, help='whether utilize mixup process')
    parser.add_argument('--mixup_alpha', default=4.0, type=float, help='mixup: beta parameter, default=4.0')
    parser.add_argument('--mixup_weight', default=1.0, type=float, help='mixup: loss weight, default=1.0')

    # -------------- for noise correction --------------
    parser.add_argument('--correct_auto', action='store_true', default=False, help='whether auto select correct_start')
    parser.add_argument('--correct_autowin', default=100, type=int, help='winnum for auto selection')
    parser.add_argument('--correct_duration', default=2000, type=int, help='duration for correction')
    parser.add_argument('--correct_start', default=2000, type=int, help = 'Start epoch to correct partial labels [default: no correct]')
    parser.add_argument('--correct_threshold_range', default='0.2,0.2', type=str, help = 'Correct threshold')
    parser.add_argument('--correct_threshold_auto', action='store_true', default=False, help = 'Whether automatically determine the threshold')
    parser.add_argument('--correct_type', default='cluster', type=str, help = 'Correct type: cluster or classify')
    parser.add_argument('--correct_update', default='case3', type=str, help = 'Correct update: none or case1 or case2 or case3')
    parser.add_argument('--correct_deletion', action='store_true', default=False, help='whether utilize deletion process')
    parser.add_argument('--augmentation_type', default='case3', type=str, help='augmemtation type from case1~case6, default=case3')

    ## -------------- optimizer parameters --------------
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_adjust', default='Case1', type=str, help='Learning rate adjust manner: Case1 or Case2.')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay (default: 1e-5).')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer for training: adam or sgd')
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether to save whole results')
    parser.add_argument('--save_root', help='where to save results', default='./savemodels', type=str, required=False)

    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    args.correct_threshold_range = [float(item) for item in args.correct_threshold_range.split(',')]
    print(args)
    cudnn.benchmark = True
    torch.set_printoptions(precision=2, sci_mode=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.gpu)
    
    print (f'====== Step1: Reading Data =======')
    if args.dataset == 'cifar10':
        args.num_class = 10
        train_loader, train_givenY, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100':
        args.num_class = 100
        train_loader, train_givenY, test_loader = load_cifar100(args)
    elif args.dataset == 'kmnist':
        args.num_class = 10
        train_loader, train_givenY, test_loader = load_kmnist(args)
    print (f'training samples: {len(train_loader.dataset)}')
    print (f'testing samples: {len(test_loader.dataset)}')

    # normalize train_givenY
    train_givenY = torch.FloatTensor(train_givenY)
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.cuda()
    print (confidence)

    # set loss functions
    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    

    print (f'====== Step2: Gaining model and optimizer =======')
    if args.encoder == 'resnet':
        model = PiCO(args, CIFAR_ResNet, pretrained=False) # pretrain is not suitable for cifar dataset
    elif args.encoder == 'preact':
        model = PiCO(args, CIFAR_PreActResNet, pretrained=False)
    model = model.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=0.9, 
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, 
                                     weight_decay=args.weight_decay)


    print (f'====== Step3: Training and Evaluation =======')
    test_accs = []
    all_labels = []
    for epoch in range(1, args.epochs+1):
        if args.lr_adjust == 'Case1':
            adjust_learning_rate_V1(args, optimizer, epoch)
        elif args.lr_adjust == 'Case2':
            adjust_learning_rate_V2(args, optimizer, epoch)
        train_save = train(args, epoch, train_loader, model, loss_fn, loss_cont_fn, optimizer)
        loss_fn.set_conf_ema_m(epoch, args)
        test_acc, test_save = test(args, epoch, test_loader, model)
        test_accs.append(test_acc)
        # automatically select args.correct_start (only change once)
        if args.correct_start == 2000 and args.correct_auto: # non-specific correct_start
            if whether_start_correct(test_accs, winnum=args.correct_autowin):
                args.correct_start = epoch
        # save results
        all_labels.append({'epoch_cls_acc':      train_save['epoch_cls_acc'],
                           'epoch_cont_acc':     train_save['epoch_cont_acc'],
                           'epoch_bingo_rate':   train_save['epoch_bingo_rate'],
                           'epoch_partial_rate': train_save['epoch_partial_rate'],
                           'epoch_test_acc':     test_acc,
                           })
        if args.savewhole and epoch % 50 == 0: # further save data which occupy much space
            all_labels[-1]['total_plabels'] = train_save['total_plabels']
            all_labels[-1]['total_dlabels'] = train_save['total_dlabels']
            all_labels[-1]['total_classfy_out'] = train_save['total_classfy_out']
            all_labels[-1]['total_cluster_out'] = train_save['total_cluster_out']
            all_labels[-1]['test_probs']  = test_save['test_probs']
            all_labels[-1]['test_preds']  = test_save['test_preds']
            all_labels[-1]['test_labels'] = test_save['test_labels']

    print (f'====== Step4: Saving =======')
    save_root = args.save_root
    if not os.path.exists(save_root): os.makedirs(save_root)

    ## gain suffix_name
    modelname = 'origin' if args.correct_start > args.epochs else 'correct'
    suffix_name = f'{args.dataset}_modelname:{modelname}_plrate:{args.partial_rate}_noiserate:{args.noise_rate}_loss:{args.loss_type}_model:{args.encoder}'
    ## gain res_name
    best_index = np.argmax(np.array(test_accs))
    bestacc = test_accs[best_index]
    res_name = f'testacc:{bestacc}'

    save_path = f'{save_root}/{suffix_name}_{res_name}_{time.time()}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        all_labels=all_labels,
                        )
