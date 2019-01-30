from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

from lfw.lfw_helper import *
from lfw.lfw_pytorch import *

from models.resnet import *

from pdb import set_trace as bp

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    # model = resnet18()
    # model.load_state_dict(torch.load("lfw/resnet18-model-arcface.pth"))
    model = resnet50()
    model.load_state_dict(torch.load("pth/resnet50_current.pth", map_location='cpu'))
    model.to(device)
    embedding_size = 512
    model.eval()

    ######## LFW dataset setup
    lfw_dir='../datasets/lfw_160'
    # lfw_dir='../lfw_160'
    lfw_pairs = 'lfw//pairs.txt'
    batch_size = 100
    num_workers = 2
    lfw_dataset = LFW(lfw_dir=lfw_dir, lfw_pairs=lfw_pairs)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ### LFW validate
    lfw_nrof_folds = 10 
    distance_metric = 0
    subtract_mean = False
    tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(model, lfw_loader, lfw_dataset, embedding_size, device,
                                                                lfw_nrof_folds, distance_metric, subtract_mean)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)