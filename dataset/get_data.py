
from __future__ import print_function
from __future__ import division

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(use_cuda, num_workers, batch_size, batch_size_test):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size_test, shuffle=True, **kwargs)

    return train_loader, test_loader
