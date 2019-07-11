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
##########################################################################
###### Arcface not Eugene  backbone_ir50_ms1m_epoch120.pth

============================================================
Validation TYPE: lfw
Accuracy: 0.99150+-0.00565
Validation rate: 0.97267+-0.01373 @ FAR=0.00133
Area Under Curve (AUC): 0.998
============================================================
============================================================
Validation TYPE: calfw
Accuracy: 0.91500+-0.04036
Validation rate: 0.34517+-0.34530 @ FAR=0.00050
Area Under Curve (AUC): 0.240
============================================================
============================================================
Validation TYPE: cplfw
Accuracy: 0.71550+-0.09986
Validation rate: 0.09717+-0.09788 @ FAR=0.00067
Area Under Curve (AUC): 0.209
============================================================
============================================================
Validation TYPE: cfp_ff
Accuracy: 0.97686+-0.00861
Validation rate: 0.91000+-0.02771 @ FAR=0.00114
Area Under Curve (AUC): 0.994
============================================================
============================================================
Validation TYPE: cfp_fp
Accuracy: 0.72129+-0.01555
Validation rate: 0.08543+-0.02215 @ FAR=0.00143
Area Under Curve (AUC): 0.791
============================================================

##########################################################################
###### Cosface Eugene IR_50_MODEL_cosface_casia_epoch51.pth
============================================================
Validation TYPE: lfw
Accuracy: 0.98483+-0.00589
Validation rate: 0.91733+-0.02546 @ FAR=0.00100
Area Under Curve (AUC): 0.998
============================================================
============================================================
Validation TYPE: calfw
Accuracy: 0.84600+-0.04564
Validation rate: 0.14983+-0.15049 @ FAR=0.00050
Area Under Curve (AUC): 0.233
============================================================
============================================================
Validation TYPE: cplfw
Accuracy: 0.73817+-0.05919
Validation rate: 0.02567+-0.02676 @ FAR=0.00050
Area Under Curve (AUC): 0.218
============================================================
============================================================
Validation TYPE: cfp_ff
Accuracy: 0.98657+-0.00351
Validation rate: 0.93086+-0.01456 @ FAR=0.00086
Area Under Curve (AUC): 0.999
============================================================
============================================================
Validation TYPE: cfp_fp
Accuracy: 0.93800+-0.01239
Validation rate: 0.69343+-0.03101 @ FAR=0.00114
Area Under Curve (AUC): 0.981
============================================================
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


def validate_forward_pass(model, lfw_loader, lfw_dataset, embedding_size, device, lfw_nrof_folds, distance_metric, subtract_mean):

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
    # embeddings = np.load('embeddings.npy')

    # np.save('embeddings_casia.npy', embeddings) 
    # embeddings = np.load('embeddings_casia.npy')

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, lfw_dataset.actual_issame, nrof_folds=lfw_nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return tpr, fpr, accuracy, val, val_std, far

#-------------------------------------------------------------
# LFW
def get_paths_issame_LFW(lfw_dir):

    lfw_images_dir = lfw_dir + '/images'
    lfw_pairs = lfw_dir + '/pairs_LFW.txt'

    # Read the file containing the pairs used for testing
    pairs = read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = get_paths(os.path.expanduser(lfw_images_dir), pairs)

    return paths, actual_issame

#-------------------------------------------------------------

# CPLFW
def get_paths_issame_CPLFW(cplfw_dir):
    cplfw_images_dir = cplfw_dir + '/images'
    cplfw_pairs = cplfw_dir + '/pairs_CPLFW.txt'
    return get_paths_issame_ca_or_cp_lfw(cplfw_images_dir, cplfw_pairs)

# CALFW
def get_paths_issame_CALFW(calfw_dir):
    calfw_images_dir = calfw_dir + '/images'
    calfw_pairs = calfw_dir + '/pairs_CALFW.txt'
    return get_paths_issame_ca_or_cp_lfw(calfw_images_dir, calfw_pairs)


def get_paths_issame_ca_or_cp_lfw(lfw_dir, lfw_pairs):

    pairs = []
    with open(lfw_pairs, 'r') as f:
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

            dir = os.path.expanduser(lfw_dir)
            path1 = os.path.join(dir, first_in_pair[0])
            path2 = os.path.join(dir, second_in_pair[0])
            paths.append(path1)
            paths.append(path2)

            if first_in_pair[1] != '0':
                actual_issame.append(True)
            else:
                actual_issame.append(False)
    
    return paths, actual_issame

#-------------------------------------------------------------
# CFP_FF and CFP_FP
def get_paths_issame_CFP(cfp_dir, type='FF'):

    pairs_list_F = cfp_dir + '/Pair_list_F.txt'
    pairs_list_P = cfp_dir + '/Pair_list_P.txt'

    path_hash_F = {}
    with open(pairs_list_F, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            path_hash_F[pair[0]] = cfp_dir + pair[1]

    path_hash_P = {}
    with open(pairs_list_P, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            path_hash_P[pair[0]] = cfp_dir + pair[1]


    paths = []
    actual_issame = []

    if type == 'FF':
        root_FF_or_FP = cfp_dir + '/Split/FF'
    else:
        root_FF_or_FP = cfp_dir + '/Split/FP'


    for subdir, _, files in os.walk(root_FF_or_FP):
        for file in files:
            filepath = os.path.join(subdir, file)

            pairs_arr = parse_dif_same_file(filepath)
            for pair in pairs_arr:
            
                first = path_hash_F[pair[0]]

                if type == 'FF':
                    second = path_hash_F[pair[1]]
                else:
                    second = path_hash_P[pair[1]]
                

                paths.append(first)
                paths.append(second)

                if file == 'diff.txt':
                    actual_issame.append(False)
                else:
                    actual_issame.append(True)

    return paths, actual_issame


def parse_dif_same_file(filepath):
    pairs_arr = []
    with open(filepath, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split(',')
            pairs_arr.append(pair)
    return pairs_arr     

#-------------------------------------------------------------

def validate_model(model, device, root_dir, type='lfw', num_workers=2, input_size=[112, 112], batch_size=100, distance_metric=1, lfw_nrof_folds=10, subtract_mean=False, print_log=False):
    """
    distance_metric = 1 #### if CenterLoss = 0, If Arcface = 1
    """
    ######## dataset setup
    if type == 'CALFW':
        paths, actual_issame = get_paths_issame_CALFW(root_dir)
    elif type == 'CPLFW':
        paths, actual_issame = get_paths_issame_CPLFW(root_dir)
    elif type == 'CFP_FF':
        paths, actual_issame = get_paths_issame_CFP(root_dir, type='FF')
    elif type == 'CFP_FP':
        paths, actual_issame = get_paths_issame_CFP(root_dir, type='FP')
    else:
        paths, actual_issame = get_paths_issame_LFW(root_dir)

    lfw_dataset = ValidateDataset(paths=paths, actual_issame=actual_issame, input_size=input_size)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('Runnning forward pass on {} images'.format(type))

    ### validate forward pass
    tpr, fpr, accuracy, val, val_std, far = validate_forward_pass(model, 
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
    # model.load_state_dict(torch.load("./pth/backbone_ir50_ms1m_epoch120.pth", map_location='cpu'))
    model.to(device)
    embedding_size = 512
    model.eval()

    # ### Validate LFW Example
    # # distance_metric = 1 #### if CenterLoss = 0, If Arcface = 1
    tpr, fpr, accuracy, val, val_std, far = validate_model(model=model, 
                                                        device=device,
                                                        root_dir='./data/lfw_112', 
                                                        type='LFW',
                                                        num_workers=2,
                                                        distance_metric=1,
                                                        print_log=True)


    ### Validate CALFW Example
    tpr, fpr, accuracy, val, val_std, far = validate_model(model=model, 
                                                        device=device,
                                                        root_dir='./data/calfw_112', 
                                                        type='CALFW',
                                                        num_workers=2,
                                                        distance_metric=1,                                                        
                                                        print_log=True)

    ### Validate CPLFW Example
    tpr, fpr, accuracy, val, val_std, far = validate_model(model=model, 
                                                        device=device, 
                                                        root_dir='./data/cplfw_112', 
                                                        type='CPLFW',
                                                        num_workers=2,
                                                        distance_metric=1,
                                                        print_log=True)


    ### Validate CFP_FF Example
    tpr, fpr, accuracy, val, val_std, far = validate_model(model=model, 
                                                        device=device, 
                                                        root_dir='./data/cfp_112', 
                                                        type='CFP_FF',
                                                        num_workers=2,
                                                        distance_metric=1,
                                                        print_log=True)

    ### Validate CFP_FP Example
    tpr, fpr, accuracy, val, val_std, far = validate_model(model=model, 
                                                        device=device, 
                                                        root_dir='./data/cfp_112', 
                                                        type='CFP_FP',
                                                        num_workers=2,
                                                        distance_metric=1,
                                                        print_log=True)
