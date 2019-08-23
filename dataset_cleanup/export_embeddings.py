
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import align.detect_face
import glob

from pdb import set_trace as bp

from six.moves import xrange
from dataset.dataset_helpers import *

import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from models.resnet import *
from models.irse import *

from helpers import *
import h5py

### Requirement: name, path, embedding

'''
person1_name
    person1_subgroup_1
        file_path    '/path/to/file1'
        embedding    [4.5, 2.1, 9.9]
    person1_subgroup_2
        file_path    '/path/to/file123'
        embedding    [84.5, 32.32, 10.1]

person2_name
    person2_subgroup_1
        file_path    '/path/to/file4444'
        embedding    [1.1, 2.1, 2.9]
    person2_subgroup_2
        file_path    '/path/to/file1123123'
        embedding    [3.0, 41.1, 56.621]

'''


"""

#################################################################################
#################################################################################
#################################################################################
COSFACE LOSS-Eugene Casia
#################################################################################

python3 dataset_cleanup/export_embeddings.py ./pth/IR_50_MODEL_cosface_casia_epoch26_lfw9895.pth ./data/embedding_test/ \
--image_size 112 \
--image_batch 5 \
--h5_name dataset.h5

"""

class FacesDataset(data.Dataset):
    def __init__(self, image_list, label_list, names_list, num_classes, image_size):
        self.image_list = image_list
        self.label_list = label_list
        self.names_list = names_list
        self.num_classes = num_classes

        self.image_size = image_size
        self.static = 0

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')

        image_data_rgb = np.asarray(data) # (112, 112, 3)

        ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=True)        
        # data = self.transforms(data)
        label = self.label_list[index]
        name = self.names_list[index]

        apsolute_path = os.path.abspath(img_path)
        return ccropped, flipped, label, name, apsolute_path

    def __len__(self):
        return len(self.image_list)

def writePerson(h5_filename, person_name, image_name, image_path, embedding):
    with h5py.File(h5_filename, 'a') as f:

        #### Person1 Folder
        person1_grp = f.create_group('person1_name')

        person1_subgroup_1 = person1_grp.create_group('person1_subgroup_1')
        person1_subgroup_1.create_dataset('embedding',  data=[4.5, 2.1, 9.9])  
        person1_subgroup_1.attrs["file_path"] = np.string_('/path/to/file1')

        person1_subgroup_2 = person1_grp.create_group('person1_subgroup_2')
        person1_subgroup_2.create_dataset('embedding',  data=[84.5, 32.32, 10.1])  
        person1_subgroup_2.attrs["file_path"] = np.string_('/path/to/file123')


def main(ARGS):
    
    np.set_printoptions(threshold=sys.maxsize)

    out_dir = ARGS.output_dir
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    train_set = get_dataset(ARGS.data_dir)
    image_list, label_list, names_list = get_image_paths_and_labels(train_set)
    faces_dataset = FacesDataset(image_list=image_list, 
                                    label_list=label_list, 
                                    names_list=names_list, 
                                    num_classes=len(train_set), 
                                    image_size=ARGS.image_size)
    loader = torch.utils.data.DataLoader(faces_dataset, batch_size=ARGS.image_batch,
                                                shuffle=False, num_workers=ARGS.num_workers)


    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(ARGS.data_dir)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    label_strings = [name for name in classes if \
       os.path.isdir(os.path.join(path_exp, name))]


    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()

    embedding_size = 512
    # emb_array = np.zeros((nrof_images, embedding_size))
    start_time = time.time()
    
########################################
    nrof_images = len(image_list)

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((0,0))

    batch_ind = 0
    with torch.no_grad():
        for i, (ccropped, flipped, label, name, apsolute_path) in enumerate(loader):

            ccropped, flipped, label = ccropped.to(device), flipped.to(device), label.to(device)

            # feats = model(data)
            feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
            
            # for j in range(len(ccropped)):
            #     # bp()
            #     dist = distance(feats_im.cpu().numpy(), feats[j].view(1,-1).cpu().numpy())
            #     # dist = distance(feats_im, feats[j])
            #     print("11111 Distance Eugene with {}  is  {}:".format(name[j], dist))

            emb = feats.cpu().numpy()
            lab = label.detach().cpu().numpy()

            # nam_array[lab] = name
            # lab_array[lab] = lab

            for j in range(len(ccropped)):
                emb_array[j+batch_ind, :] = emb[j, :]

                bp()
                writePerson(ARGS.h5_name, person_name, image_name, image_path, embedding)

            lab_array = np.append(lab_array,lab)
            
            # print("\n")
            # for j in range(len(ccropped)):
            #     dist = distance(feats_im.cpu().numpy(), np.expand_dims(emb_array[j+batch_ind], axis=0))
            #     # dist = distance(feats_im, feats[j])
            #     print("22222 Distance Eugene with {}  is  {}:".format(name[j], dist))
            # print("\n")

            bp()
            batch_ind += len(ccropped)

            if i % 10 == 9:
                print('.', end='')
                sys.stdout.flush()
        print('')

    run_time = time.time() - start_time
    print('Run time: ', run_time)

    print("Exporting All embeddings")
    #   export emedings and labels
    # np.save(out_dir + ARGS.embeddings_name, emb_array)
    # np.save(out_dir + ARGS.labels, lab_array)

    # label_strings = np.array(label_strings)
    # np.save(out_dir + ARGS.labels_strings, label_strings[label_list])



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='pth model file')
    parser.add_argument('data_dir', type=str, help='Directory containing images.')
    parser.add_argument('--output_dir', type=str, help='Dir where to save all embeddings and demo images', default='output_arrays/')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_batch', type=int, help='Number of images stored in memory at a time. Default 64.', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)

    parser.add_argument('--h5_name', type=str, help='h5 file name', default='dataset.h5')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
