import os
import sys
import pickle 
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.utils_algo import *
from utils.randaugment import RandomAugment


def load_cifar10(args, transform):
    #######################################################
    print ('obtain train_loader')
    ## train_loader: (data, labels), only read data (target data: (60000, 32, 32, 3))
    temp_train = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=True, download=True)
    data_train, dlabels_train = temp_train.data, temp_train.targets # (50000, 32, 32, 3)
    assert np.min(dlabels_train) == 0, f'min(dlabels) != 0'

    ## train_loader: train_givenY
    dlabels_train = np.array(dlabels_train).astype('int')
    num_sample = len(dlabels_train)
    train_givenY = generate_uniform_cv_candidate_labels(dlabels_train, args.partial_rate) ## generate partial dlabels
    print('Average candidate num: ', np.mean(np.sum(train_givenY, axis=1)))
    bingo_rate = np.sum(train_givenY[np.arange(num_sample), dlabels_train] == 1.0) / num_sample
    print('Average bingo rate: ', bingo_rate)
    train_givenY = generate_noise_labels(dlabels_train, train_givenY, args.noise_rate)
    bingo_rate = np.sum(train_givenY[np.arange(num_sample), dlabels_train] == 1.0) / num_sample
    print('Average noise rate: ', 1 - bingo_rate)
 
    ## train_loader: train_givenY->plabel
    dlabels_train = np.array(dlabels_train).astype('float')
    train_givenY = np.array(train_givenY).astype('float')
    plabels_train = (train_givenY!=0).astype('float')

    partial_matrix_dataset = Augmentention(data_train, plabels_train, dlabels_train, transform)
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True)

    #######################################################
    print ('obtain test_loader')
    temp_test = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True)
    data_test, dlabels_test = temp_test.data, temp_test.targets # (50000, 32, 32, 3)
    assert np.min(dlabels_test) == 0, f'min(dlabels) != 0'

    ## (data, dlabels) -> test_loader
    test_dataset = Augmentention(data_test, dlabels_test, dlabels_test, transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False)

    return partial_matrix_train_loader, train_givenY, test_loader



class Augmentention(Dataset):
    def __init__(self, images, plabels, dlabels, transforms):
        self.images = images
        self.plabels = plabels
        self.dlabels = dlabels
        self.transforms = transforms

    def __len__(self):
        return len(self.dlabels)
        
    def __getitem__(self, index):
        each_image = self.images[index]
        each_image = self.transforms(each_image)
        each_plabel = self.plabels[index]
        each_dlabel = self.dlabels[index]
        return each_image, each_plabel, each_dlabel, index


