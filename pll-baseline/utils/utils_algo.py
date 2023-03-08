import os
import cv2
import math
import time
import glob
import tqdm
import random
import shutil
import argparse
import builtins
import warnings
import torchvision
import numpy as np

# learning rate adjust for each epoch
def adjust_learning_rate_V1(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (0.1 ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_V2(args, optimizer, epoch):
    lr = args.lr * args.decayrate ** (epoch // args.decaystep)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def auto_correct_threshold(args):
    q_new = args.partial_rate + args.noise_rate / (args.num_class-1)
    k = (q_new * args.noise_rate) / ((1-q_new)*(1-args.noise_rate))
    return k

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


## input: root;
## output: data (N, 32, 32, 3) without labels， 读入图片uint8格式，数值范围是0-255
def read_all_imagepath(root):
    image_path_whole = []

    for root_subpath in glob.glob(root+'/*'):
        if os.path.isdir(root_subpath):
            image_path_whole.extend(read_all_imagepath(root_subpath))
        elif os.path.isfile(root_subpath) and root_subpath.rsplit('.', 1)[-1] in ['JPEG', 'jpg']:
            image_path_whole.append(root_subpath)

    return image_path_whole

def read_all_images(root):
    image_whole = []

    image_path_whole = read_all_imagepath(root)
    for img_path in tqdm.tqdm(image_path_whole):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (32, 32))
        image_whole.append(image)
    image_whole = np.array(image_whole)
    print (f'sample number: {len(image_whole)}')

    np.savez_compressed(f'{root}.npz',
                        image_whole=image_whole)

def read_ood_data(args):
    if args.ood_dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(root='../dataset/MNIST', train=True, download=True).data
        test_data = torchvision.datasets.MNIST(root='../dataset/MNIST', train=False, download=True).data
        whole_data = np.concatenate([train_data, test_data], axis=0)
        whole_data = whole_data[:,:,:,np.newaxis]
        whole_data = np.tile(whole_data, (1, 1, 1, 3)) # [N, 28, 28, 3]
        whole_data = np.array([cv2.resize(data, (32, 32)) for data in whole_data if 1==1]) # [N, 32, 32, 3]
    elif args.ood_dataset == 'kmnist':
        train_data = torchvision.datasets.KMNIST(root='../dataset/KMNIST', train=True, download=True).data
        test_data = torchvision.datasets.KMNIST(root='../dataset/KMNIST', train=False, download=True).data
        whole_data = np.concatenate([train_data, test_data], axis=0)
        whole_data = whole_data[:,:,:,np.newaxis]
        whole_data = np.tile(whole_data, (1, 1, 1, 3))
        whole_data = np.array([cv2.resize(data, (32, 32)) for data in whole_data if 1==1])
    elif args.ood_dataset == 'fashion':
        train_data = torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST', train=True, download=True).data
        test_data = torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST', train=False, download=True).data
        whole_data = np.concatenate([train_data, test_data], axis=0)
        whole_data = whole_data[:,:,:,np.newaxis]
        whole_data = np.tile(whole_data, (1, 1, 1, 3))
        whole_data = np.array([cv2.resize(data, (32, 32)) for data in whole_data if 1==1])
    elif args.ood_dataset == 'tinyimagenet':
        whole_data = np.load('../dataset/tinyimagenet.npz')['image_whole']
    elif args.ood_dataset == 'texture':
        whole_data = np.load('../dataset/texture.npz')['image_whole']
    elif args.ood_dataset == 'places365':
        whole_data = np.load('../dataset/places365.npz')['image_whole']
    else:
        assert 1==0, 'this dataset does not exist!!'
    assert np.shape(whole_data)[-1] == 3
    # print (f'sample number: {len(whole_data)}')
    return whole_data # must be [N, width, height, 3]


# data_train: [N, 32, 32, 3]
def calculate_mean_std(data_train):

    data_train = data_train / 255.0
    sample_number = data_train.shape[0]
    feature_dim = data_train.shape[-1]
    data_train = data_train.reshape((sample_number, -1, feature_dim)) # [N, 32*32, 3]
    mean_data = np.mean(data_train, axis=1) # [N, 3]

    mean_value = np.mean(mean_data, axis=0)
    std_value = np.std(mean_data, axis=0)
    print (f'statistic mean: {mean_value}')
    print (f'statistic std: {std_value}')
    return mean_value, std_value


def read_ood_data_temp():
    for ood_dataset in ['mnist', 'kmnist', 'fashion', 'tinyimagenet', 'texture', 'places365']:
        if ood_dataset == 'mnist':
            train_data = torchvision.datasets.MNIST(root='../dataset/MNIST', train=True, download=True).data
            test_data = torchvision.datasets.MNIST(root='../dataset/MNIST', train=False, download=True).data
            whole_data = np.concatenate([train_data, test_data], axis=0)
            whole_data = whole_data[:,:,:,np.newaxis]
            whole_data = np.tile(whole_data, (1, 1, 1, 3)) # [N, 28, 28, 3]
            whole_data = np.array([cv2.resize(data, (32, 32)) for data in whole_data if 1==1]) # [N, 32, 32, 3]
        elif ood_dataset == 'kmnist':
            train_data = torchvision.datasets.KMNIST(root='../dataset/KMNIST', train=True, download=True).data
            test_data = torchvision.datasets.KMNIST(root='../dataset/KMNIST', train=False, download=True).data
            whole_data = np.concatenate([train_data, test_data], axis=0)
            whole_data = whole_data[:,:,:,np.newaxis]
            whole_data = np.tile(whole_data, (1, 1, 1, 3))
            whole_data = np.array([cv2.resize(data, (32, 32)) for data in whole_data if 1==1])
        elif ood_dataset == 'fashion':
            train_data = torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST', train=True, download=True).data
            test_data = torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST', train=False, download=True).data
            whole_data = np.concatenate([train_data, test_data], axis=0)
            whole_data = whole_data[:,:,:,np.newaxis]
            whole_data = np.tile(whole_data, (1, 1, 1, 3))
            whole_data = np.array([cv2.resize(data, (32, 32)) for data in whole_data if 1==1])
        elif ood_dataset == 'tinyimagenet':
            whole_data = np.load('../dataset/tinyimagenet.npz')['image_whole']
        elif ood_dataset == 'texture':
            whole_data = np.load('../dataset/texture.npz')['image_whole']
        elif ood_dataset == 'places365':
            whole_data = np.load('../dataset/places365.npz')['image_whole']
        else:
            assert 1==0, 'this dataset does not exist!!'
        assert np.shape(whole_data)[-1] == 3
        print (f'ood dataset: {ood_dataset}')
        print (f'sample number: {len(whole_data)}')
        # return whole_data # must be [N, width, height, 3]


if __name__ == '__main__':
    import fire
    fire.Fire()