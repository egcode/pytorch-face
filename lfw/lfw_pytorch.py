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
from models.resnet import *

from pdb import set_trace as bp

class LFW(data.Dataset):
    
    def __init__(self, lfw_dir, lfw_pairs):

        # Read the file containing the pairs used for testing
        pairs = read_pairs(os.path.expanduser(lfw_pairs))

        # Get the paths for the corresponding images
        self.paths, self.actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)
        self.nrof_embeddings = len(self.actual_issame)*2  # nrof_pairs * nrof_images_per_pair
        self.labels_array = np.arange(0,self.nrof_embeddings)

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path)
        data = img.convert('RGB')
        data = self.transforms(data)
        label = self.labels_array[index]
        return data.float(), label

    def __len__(self):
        return len(self.paths)


def lfw_validate_model(lfw_dir, lfw_pairs, batch_size, num_workers, model, embedding_size, device):

    model.eval()
    model.to(device)

    lfw_dataset = LFW(lfw_dir=lfw_dir,
                     lfw_pairs=lfw_pairs)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
    print('Runnning forward pass on LFW images')

    lfw_nrof_folds = 10 
    distance_metric = 0
    subtract_mean = False

    nrof_images = lfw_dataset.nrof_embeddings 

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i, (data, label) in enumerate(lfw_loader):

        data, label = data.to(device), label.to(device)

        emb = model(data).detach().cpu().numpy()
        lab = label.detach().cpu().numpy()

        lab_array[lab] = lab
        emb_array[lab, :] = emb

        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = emb_array

    # np.save('embeddings.npy', embeddings) 
    # embeddings = np.load('lfw/embeddings.npy')
    
    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, lfw_dataset.actual_issame, nrof_folds=lfw_nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return tpr, fpr, accuracy, val, val_std, far
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    model = resnet18()
    model.load_state_dict(torch.load("lfw/resnet18-model-arcface.pth"))
    embedding_size = model.fc5.out_features

    ######## LFW setup
    lfw_dir='../Computer-Vision/datasets/lfw_160'
    lfw_pairs = 'lfw//pairs.txt'
    batch_size = 100
    num_workers = 2
    tpr, fpr, accuracy, val, val_std, far = lfw_validate_model(lfw_dir, lfw_pairs, batch_size, num_workers, model, embedding_size, device)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)
