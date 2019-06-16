from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
#################################################################################
#################################################################################
#################################################################################
CENTER LOSS
#################################################################################

python3 demo_compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_path ./data/test_image.png \
--embeddings_premade ./output_arrays/embeddings_center_1.npy \
--label_string_center ./output_arrays/label_strings_center_1.npy \
--labels_center ./output_arrays/labels_center_1.npy \
--distance_metric 0


'''
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
import cv2

from imutils.video import VideoStream
import imutils
import time

import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
from PIL import Image

from models.resnet import *
from models.irse import *
from helpers import *
from pdb import set_trace as bp

def main(ARGS):

    ###### IMAGE
    img_path = ARGS.image_path
    img = Image.open(img_path)
    image_data = img.convert('RGB')
    image_data_rgb = np.asarray(image_data) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=False)
    # image_data.save('pilllllllll.png')

    ###### EMBEDDINGS
    embeddings_premade = np.load(ARGS.embeddings_premade, allow_pickle=True)
    label_string_center = np.load(ARGS.label_string_center, allow_pickle=True)
    labels_center = np.load(ARGS.labels_center, allow_pickle=True)

    
    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()
    #########################################

    with torch.no_grad():
        feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
        feats = feats.cpu().numpy()


    #########################################

    nrof_premade = embeddings_premade.shape[0]
    all_results_dict = {}

    for j in range(nrof_premade):
        # dist = np.sqrt(np.sum(np.square(np.subtract(face.embedding, embeddings_premade[j,:]))))
        dist = distance(feats, embeddings_premade[j,:], ARGS.distance_metric)

        # dist = distance(face.embedding, embeddings_premade[j,:].reshape((1, 512)), ARGS.distance_metric)
        print("Distance with {}: {}".format(label_string_center[j], dist))

        label = label_string_center[j]
        if label in all_results_dict: # if label value in dictionary
            arr = all_results_dict.get(label)
            arr.append(dist)
        else:
            all_results_dict[label] = [dist]
        # print("candidate: " + str(i) + " distance: " + str(dist) + " with " + label_string_center[j])



    print("======EMBEDDINGS ALL RESULTS===========")
    print(all_results_dict)
    print("=================")
    

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pth model file')
    parser.add_argument('--image_path', type=str, help='image to compare')
    parser.add_argument('--embeddings_premade', type=str, help='Premade embeddings array .npy format')
    parser.add_argument('--label_string_center', type=str, help='Premade label strings array .npy format')
    parser.add_argument('--labels_center', type=str, help='Premade labels integers array .npy format')
    parser.add_argument('--distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

