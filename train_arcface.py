from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
from sklearn import metrics

from losses.Arcface_loss import Arcface_loss
from dataset.get_data import get_data
from models.net import Net
from models.resnet import *

from lfw.lfw_pytorch import *
from lfw.lfw_helper import *

print("Pytorch version:  " + str(torch.__version__))
use_cuda = torch.cuda.is_available()
print("Use CUDA: " + str(use_cuda))

from pdb import set_trace as bp

BATCH_SIZE = 11
FEATURES_DIM = 512
NUM_OF_CLASSES = 10
BATCH_SIZE_TEST = 1000
EPOCHS = 20
LOG_INTERVAL = 10
NUM_WORKERS = 2
# MODEL_TYPE = 'resnet18_face'
MODEL_TYPE = 'resnet18'
# MODEL_TYPE = 'resnet34'
# MODEL_TYPE = 'resnet50'


def train(model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        features = model(data)
        logits = loss_arcface(features, target)
        loss = loss_softmax(logits, target)

        _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer_nn.zero_grad()
        optimzer_arcface.zero_grad()

        loss.backward()

        optimizer_nn.step()
        optimzer_arcface.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_softmax, loss_arcface):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            feats = model(data)
            logits = loss_arcface(feats, target)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target.data).sum()

    print('\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    

def validate_lfw(model, device, epoch):
    model.eval()
    embedding_size = model.fc5.out_features

    ######## LFW setup
    lfw_dir='../Computer-Vision/datasets/lfw_160'
    lfw_pairs = 'lfw//pairs.txt'
    lfw_batch_size = 100
    tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(lfw_dir, lfw_pairs, lfw_batch_size, NUM_WORKERS, model, embedding_size, device)

    print('\nEpoch: '+str(epoch))
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)
    
###################################################################


if __name__ == '__main__':
    
    device = torch.device("cuda" if use_cuda else "cpu")

    ####### Data setup
    data_dir = '../Computer-Vision/datasets/CASIA-WebFace_160'
    train_loader, test_loader = get_data(data_dir, device, NUM_WORKERS, BATCH_SIZE, BATCH_SIZE_TEST)
        
    ####### Model setup
    if MODEL_TYPE == 'resnet18':
        model = resnet18()
    elif MODEL_TYPE == 'resnet34':
        model = resnet34()
    elif MODEL_TYPE == 'resnet50':
        model = resnet50()

    # model = Net(features_dim=FEATURES_DIM)
    model = model.to(device)

    loss_softmax = nn.CrossEntropyLoss().to(device)
    loss_arcface = Arcface_loss(num_classes=len(train_loader.dataset.classes), feat_dim=FEATURES_DIM, device=device).to(device)

    # optimzer nn
    optimizer_nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    sheduler_nn = lr_scheduler.StepLR(optimizer_nn, 20, gamma=0.1)

    # optimzer cosface or arcface
    optimzer_arcface = optim.SGD(loss_arcface.parameters(), lr=0.01)
    sheduler_arcface = lr_scheduler.StepLR(optimzer_arcface, 20, gamma=0.1)


    for epoch in range(1, EPOCHS + 1):
        sheduler_nn.step()
        sheduler_arcface.step()

        train(model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, epoch)
        test(model, device, test_loader, loss_softmax, loss_arcface)
        validate_lfw(model, device, epoch)
        
    torch.save(model.state_dict(),"resnet18-model-arcface.pth")        
    torch.save(loss_arcface.state_dict(),"resnet18_loss-arcface.pth")        
