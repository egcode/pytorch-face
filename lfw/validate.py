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
from models.irse import *

from pdb import set_trace as bp

"""
----------------------------------------
TEST: ---  backbone_ir50_ms1m_epoch120.pth
Accuracy: 0.99150+-0.00565
Validation rate: 0.97267+-0.01373 @ FAR=0.00133
Area Under Curve (AUC): 0.998
----------------------------------------

----------------------------------------
TEST: ---  IR_50_MODEL_cosface_casia_epoch51.pth
Accuracy: 0.98483+-0.00589
Validation rate: 0.91733+-0.02546 @ FAR=0.00100
Area Under Curve (AUC): 0.998
----------------------------------------

"""

class ValidateDataset(data.Dataset):
    
    def __init__(self, paths, actual_issame, input_size):
    
        self.paths = paths
        self.actual_issame = actual_issame

        self.nrof_embeddings = len(self.actual_issame)*2  # nrof_pairs * nrof_images_per_pair
        self.labels_array = np.arange(0,self.nrof_embeddings)

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        self.transforms = T.Compose([
            T.Resize(input_size),
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


def validate_model(model, lfw_loader, lfw_dataset, embedding_size, device, lfw_nrof_folds, distance_metric, subtract_mean):
    print('Runnning forward pass on LFW images')

    nrof_images = lfw_dataset.nrof_embeddings

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    with torch.no_grad():
        for i, (data, label) in enumerate(lfw_loader):

            data, label = data.to(device), label.to(device)

            feats = model(data)
            emb = feats.cpu().numpy()
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

#-------------------------------------------------------------
def get_paths_issame_lfw():

    lfw_dir='./data/lfw_112/images'
    lfw_pairs = './data/lfw_112/pairs_LFW.txt'

    # Read the file containing the pairs used for testing
    pairs = read_pairs(os.path.expanduser(lfw_pairs))
    bp()
    # Get the paths for the corresponding images
    paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)

    bp()
    return paths, actual_issame

#-------------------------------------------------------------

def get_paths_issame_calfw():

    calfw_dir='./data/calfw_112/images'
    calfw_pairs = './data/calfw_112/pairs_CALFW.txt'

    # Read the file containing the pairs used for testing
    # pairs = read_pairs(os.path.expanduser(calfw_pairs))

    pairs = []
    with open(calfw_pairs, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs.append(pair)
    arr = np.array(pairs)

    paths = []
    actual_issame = []
    for count, person in enumerate(arr, 1): # Start counting from 1
        if count % 2 == 0:
            first_in_pair = arr[count-2]
            second_in_pair = person

            dir = os.path.expanduser(calfw_dir)
            path1 = os.path.join(dir, first_in_pair[0])
            path2 = os.path.join(dir, second_in_pair[0])
            paths.append(path1)
            paths.append(path2)

            if first_in_pair[1] != '0':
                actual_issame.append(True)
            else:
                actual_issame.append(False)

               
            # print("\nPair num: {} first: {}   second: {}".format(count/2, first_in_pair, second_in_pair))
            # print("\Actual_issame: {}".format(actual_issame))
            
            # bp()

    # bp()
    # Get the paths for the corresponding images
    # paths, actual_issame = get_paths(os.path.expanduser(calfw_dir), pairs)

    bp()
    return paths, actual_issame

#-------------------------------------------------------------

def validate_type(model, device, type='lfw', num_workers=2, input_size=[112, 112], batch_size=100, distance_metric=1, lfw_nrof_folds=10, subtract_mean=False, print_log=False):
    """
    distance_metric = 1 #### if CenterLoss = 0, If Arcface = 1
    """
    ######## dataset setup
    if type == 'calfw':
        paths, actual_issame = get_paths_issame_calfw()
    else:
        paths, actual_issame = get_paths_issame_lfw()

    lfw_dataset = ValidateDataset(paths=paths, actual_issame=actual_issame, input_size=input_size)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ### validate
    tpr, fpr, accuracy, val, val_std, far = validate_model(model, 
                                                        lfw_loader, 
                                                        lfw_dataset, 
                                                        embedding_size, 
                                                        device,
                                                        lfw_nrof_folds, 
                                                        distance_metric, 
                                                        subtract_mean)


    if print_log == True:
        print("=" * 60)
        print("Validation TYPE: {}".format(type))
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        # print('Equal Error Rate (EER): %1.3f' % eer)
        print("=" * 60)

    return tpr, fpr, accuracy, val, val_std, far

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    model = IR_50([112, 112])
    model.load_state_dict(torch.load("./pth/IR_50_MODEL_cosface_casia_epoch51.pth", map_location='cpu'))
    model.to(device)
    embedding_size = 512
    model.eval()

    # ### Validate LFW Example
    # # distance_metric = 1 #### if CenterLoss = 0, If Arcface = 1
    # tpr, fpr, accuracy, val, val_std, far = validate_type(model=model, 
    #                                                     device=device, 
    #                                                     type='lfw',
    #                                                     num_workers=2,
    #                                                     print_log=True)



    ### Validate CALFW Example
    tpr, fpr, accuracy, val, val_std, far = validate_type(model=model, 
                                                        device=device, 
                                                        type='calfw',
                                                        num_workers=2,
                                                        print_log=True)
