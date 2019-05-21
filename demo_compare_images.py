from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''

##### NOT SAME        distance = [0.5334694]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Liuba/IMG_0179.png \
--image_two_path ./data/golovan_160/Julia/0001.png

##### NOT SAME        distance = [0.49472308]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Alex/haweF.png \
--image_two_path ./data/golovan_160/Julia/0001.png




##### SAME             distance = [0.3180063]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0001.png

##### SAME             distance = [0.50955397]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0004.png

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
# from six.moves import xrange
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


from pdb import set_trace as bp

def main(ARGS):

    normalize = T.Normalize(mean=[0.5], std=[0.5])
    transforms = T.Compose([
        T.Resize([ARGS.image_size,ARGS.image_size]),
        T.ToTensor(),
        normalize
    ])

    ###### IMAGE 1111
    img_path1 = ARGS.image_one_path
    img1 = Image.open(img_path1)
    image_data1 = img1.convert('RGB')
    image_data1 = transforms(image_data1)
    image_data1 = image_data1.reshape(1, 3, ARGS.image_size, ARGS.image_size)

    ###### IMAGE 2222
    img_path2 = ARGS.image_two_path
    img2 = Image.open(img_path2)
    image_data2 = img2.convert('RGB')
    image_data2 = transforms(image_data2)
    image_data2 = image_data2.reshape(1, 3, ARGS.image_size, ARGS.image_size)


    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()
    #########################################


    with torch.no_grad():
        feats_1 = model(torch.tensor(image_data1))
        feats_1 = feats_1.cpu().numpy()

        feats_2 = model(torch.tensor(image_data2))
        feats_2 = feats_2.cpu().numpy()
    dist = distance(feats_1, feats_2, 1) ## Distance based on cosine similarity

    print("======DISTANCE===========")
    print(dist)
    print("=================")
    # bp()

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
    parser.add_argument('--image_one_path', type=str, help='ONE')
    parser.add_argument('--image_two_path', type=str, help='TWO')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

