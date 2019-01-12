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
import argparse
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
DATA_DIR = '../Computer-Vision/datasets/CASIA-WebFace_160'

# MODEL_TYPE = 'resnet18_face'
MODEL_TYPE = 'resnet18'
# MODEL_TYPE = 'resnet34'
# MODEL_TYPE = 'resnet50'
LFW_DIR='../Computer-Vision/datasets/lfw_160'
LFW_PAIRS = 'lfw//pairs.txt'
LFW_BATCH_SIZE = 100

MODEL_SAVE_INTERVAL = 1
TEST_INTERVAL = 1
LFW_INTERVAL = 1

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


def test(model, device, test_loader, loss_softmax, loss_arcface, epoch):
    if epoch % TEST_INTERVAL == 0 or epoch == EPOCHS:
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

def validate_lfw(args, model, lfw_loader, lfw_dataset, device, epoch):
    if epoch % LFW_INTERVAL == 0 or epoch == EPOCHS:
        model.eval()
        embedding_size = model.fc5.out_features

        # tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(model, lfw_loader, lfw_dataset, embedding_size, device)

        # lfw_nrof_folds = 10 
        # distance_metric = 0
        # subtract_mean = False
        tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(model, lfw_loader, lfw_dataset, embedding_size, device,
                                                                    args.lfw_nrof_folds, args.lfw_distance_metric, args.lfw_subtract_mean)

        print('\nEpoch: '+str(epoch))
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        # print('Equal Error Rate (EER): %1.3f' % eer)


def save_model(model, type, epoch):
    # save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    # torch.save(model.state_dict(), save_name)
    if epoch % MODEL_SAVE_INTERVAL == 0 or epoch == EPOCHS:
        save_name = os.path.join('checkpoints', type + '_' + str(epoch) + '.pth')
        print("Save Model name: " + str(save_name))
###################################################################


def main(args):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    ####### Data setup
    train_loader, test_loader = get_data(DATA_DIR, device, NUM_WORKERS, BATCH_SIZE, BATCH_SIZE_TEST)

    ######## LFW Data setup
    lfw_dataset = LFW(lfw_dir=LFW_DIR, lfw_pairs=LFW_PAIRS)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=LFW_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
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
        bp()
        # train(model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, epoch)
        # test(model, device, test_loader, loss_softmax, loss_arcface, epoch)
        validate_lfw(args, model, lfw_loader, lfw_dataset, device, epoch)
        # save_model(model, MODEL_TYPE, epoch)

    # torch.save(model.state_dict(),"resnet18-model-arcface.pth")        
    # torch.save(loss_arcface.state_dict(),"resnet18_loss-arcface.pth")        


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='lfw//pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='../Computer-Vision/datasets/lfw_160')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true', default=False)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
