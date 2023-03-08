import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import pickle
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_correct_threshold(args, epoch):
    ## epoch >= args.correct_start
    start = args.correct_threshold_range[0]
    end = args.correct_threshold_range[1]
    correct_threshold = (epoch - args.correct_start) / (args.epochs - args.correct_start) * (end - start) + start
    return correct_threshold


def auto_correct_threshold(args):
    q_new = args.partial_rate + args.noise_rate / (args.num_class-1)
    k = (q_new * args.noise_rate) / ((1-q_new)*(1-args.noise_rate))
    return k


## lr: move args.lr to args.lr * 1e-3
def adjust_learning_rate_V1(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (0.1 ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_V2(args, optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))



def generate_uniform_cv_candidate_labels(labels, partial_rate=0.1):

    K = int(np.max(labels) - np.min(labels) + 1) # 10
    n = len(labels) # 50000

    partialY = np.zeros((n, K))
    partialY[np.arange(n), labels] = 1.0

    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=partial_rate
    # print(transition_matrix)
    '''
    transition_matrix = 
        [[1.  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 1.  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 1.  0.5 0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 1.  0.5 0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 1.  0.5 0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 1.  0.5 0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 1.  0.5 0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.  0.5 0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.  0.5]
         [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1. ]]
    '''
    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[labels[j], jj]:
                partialY[j, jj] = 1.0

    return partialY

## add noise to partialY
def generate_noise_labels(labels, partialY, noise_rate=0.0):

    partialY_new = [] # must define partialY_new
    for ii in range(len(labels)):
        label = labels[ii]
        plabel =  partialY[ii]
        noise_flag = (random.uniform(0, 1) <= noise_rate) # whether add noise to label
        if noise_flag:
            ## random choose one idx not in plabel
            houxuan_idx = []
            for ii in range(len(plabel)):
                if plabel[ii] == 0: houxuan_idx.append(ii)
            if len(houxuan_idx) == 0: # all category in partial label
                partialY_new.append(plabel)
                continue
            ## add noise in partial label
            newii = random.randint(0, len(houxuan_idx)-1)
            idx = houxuan_idx[newii]
            assert plabel[label] == 1, f'plabel[label] != 1'
            assert plabel[idx]   == 0, f'plabel[idx]   != 0'
            plabel[label] = 0
            plabel[idx] = 1
            partialY_new.append(plabel)
        else:
            partialY_new.append(plabel)
    partialY_new = np.array(partialY_new)
    return partialY_new




def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100'

    meta = unpickle('data/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("Finish Generating Candidate Label Sets!\n")
    return partialY


# def whether_start_correct(results, winnum, margin):

#     if len(results) <= winnum:
#         return False

#     ## calculate delta_results
#     delta_results = []
#     for epoch in range(1, len(results)):
#         value = results[epoch] - results[epoch-1]
#         delta_results.append(value)

#     ## gain mean values
#     win_results = delta_results[-winnum:]
#     meanvalue = np.mean(win_results)
#     if meanvalue < margin:
#         return True
#     else:
#         return False


# results: online test_accs
# winnum:  window size for mean values
def whether_start_correct(results, winnum):
 
    if len(results) < winnum: 
        return False

    ## 平滑test_acc曲线，每个点是周围领域内点的平均值
    half_win = int(winnum/2)
    avg_results = []
    for epoch in range(len(results)):
        left_margin  = max(0, epoch-half_win)
        right_margin = min(len(results)-1, epoch+half_win)
        win_select = results[left_margin:right_margin]
        avg_results.append(np.mean(win_select))

    ## 计算delta值，找到有可能是最大值的位置（如果增长率趋向于0，就可能到极值点了）
    delta_results = []
    for epoch in range(1, len(avg_results)):
        value = avg_results[epoch] - avg_results[epoch-1]
        delta_results.append(value)

    ## 如果若干epoch内的delta都趋向于0，那么就真的到极值点了（防止结果跳变）
    win_results = delta_results[-10:]
    meanvalue = np.mean(win_results)
    if meanvalue < 1e-6:
        return True
    else:
        return False


def read_data(dataset, train):
    if dataset == 'cifar10':
        data = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=train, download=True)
    elif dataset == 'cifar100':
        data = torchvision.datasets.CIFAR100(root='../dataset/CIFAR100', train=train, download=True)
    elif dataset == 'mnist':
        data = torchvision.datasets.MNIST(root='../dataset/MNIST', train=train, download=True)
    elif dataset == 'kmnist':
        data = torchvision.datasets.KMNIST(root='../dataset/KMNIST', train=train, download=True)
    elif dataset == 'fashion':
        data = torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST', train=train, download=True)
    return data