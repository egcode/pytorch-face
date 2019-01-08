import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
from pdb import set_trace as bp

from losses.Arcface_loss import Arcface_loss
from models.net import Net

print("Pytorch version:  " + str(torch.__version__))
use_cuda = torch.cuda.is_available()
print("Use CUDA: " + str(use_cuda))


BATCH_SIZE = 100
FEATURES_DIM = 3
NUM_OF_CLASSES = 10
BATCH_SIZE_TEST = 1000
EPOCHS = 20
LOG_INTERVAL = 10
        

def train(model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        features, _ = model(data)
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

            feats, _ = model(data)
            logits = loss_arcface(feats, target)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target.data).sum()

    print('\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    

###################################################################

torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")

####### Data setup

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True, **kwargs)

####### Model setup

model = Net().to(device)
loss_softmax = nn.CrossEntropyLoss().to(device)
loss_arcface = Arcface_loss(num_classes=10, feat_dim=FEATURES_DIM, device=device).to(device)

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

torch.save(model.state_dict(),"mnist_cnn-arcface.pt")        
torch.save(loss_arcface.state_dict(),"mnist_loss-arcface.pt")        
