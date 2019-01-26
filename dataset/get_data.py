from __future__ import print_function
from __future__ import division

import torch
import torchvision
from torch.utils.data import DataLoader
# import numpy as np
from torchvision import datasets, models, transforms
import os
import cv2
from PIL import Image

from pdb import set_trace as bp


TRAIN = 'train'
TEST = 'test'

def get_data(data_dir, device, num_workers, batch_size, batch_size_test):

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(160, scale=(0.7, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        TEST: transforms.Compose([
            transforms.RandomResizedCrop(160, scale=(0.7, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in [TRAIN, TEST]}

    trainloader = torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(image_datasets[TEST], batch_size=batch_size_test,
                                                shuffle=True, num_workers=num_workers)

    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # class_names = image_datasets[TRAIN].classes #List of the class names.
    # class_to_idx = image_datasets[TRAIN].class_to_idx #Dict with items (class_name, class_index).

    total_train_imgs = image_datasets[TRAIN].imgs #List of (image path, class_index) tuples
    # total_test_imgs = image_datasets[TEST].imgs #List of (image path, class_index) tuples


    # print("\nClass names: " + str(class_names))
    # print("\nclass_to_idx: " + str(class_to_idx))

    # current_image = 0

    # for batch_idx, (data, target) in enumerate(trainloader):
    #     data, target = data.to(device), target.to(device)

    #     for ind, (image) in enumerate(data):
    #         print('image: [{}/{} ({:.0f}%)]'.format(
    #             current_image, len(total_train_imgs),
    #             100. * current_image / len(total_train_imgs)))
    #         current_image += 1

    # print("\nTrain Images COUNT: " + str(len(total_train_imgs)))
    # print("\n")

    return trainloader, testloader
