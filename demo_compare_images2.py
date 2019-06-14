from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''

##### NOT SAME        without extract_feature distance = [0.5334694] 
#                     with extract_feature distance = [0.5138081]
python3 demo_compare_images2.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Liuba/IMG_0179.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.49472308]
#                     with extract_feature distance = [0.51892143]
python3 demo_compare_images2.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Alex/haweF.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.456529]
#                     with extract_feature distance = [0.48116595]
python3 demo_compare_images2.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Eugene/hawfd.png \
--image_two_path ./data/golovan_160/Julia/IMG_1922.png \
--distance_metric 1




##### SAME            without extract_feature distance = [0.3180063]
#                     with extract_feature distance = [0.30892584]

python3 demo_compare_images2.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.50955397]
#                     with extract_feature distance = [0.49133044]
python3 demo_compare_images2.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0004.png \
--distance_metric 1


#################################################################################
##### NOT SAME       without extract_feature distance = [0.2699489]
#                     with extract_feature distance = [1.1721498]
python3 demo_compare_images2.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Liuba/IMG_0179.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 0

##### NOT SAME       without extract_feature distance = [0.21732879]
#                     with extract_feature distance = [1.2014594]
python3 demo_compare_images2.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Alex/haweF.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 0



##### SAME            without extract_feature distance = [0.12518325]
#                     with extract_feature distance = [0.58154273]
python3 demo_compare_images2.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 0

##### SAME            without extract_feature distance = [0.21716458]
#                     with extract_feature distance = [0.8795128]
python3 demo_compare_images2.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0004.png \
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





import torch
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_root, backbone, device, tta = True):
    # pre-requisites
    assert(os.path.exists(img_root))
    print('Testing Data Root:', img_root)

    # load image
    img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image
    ccropped = ccropped[...,::-1] # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())
            
#     np.save("features.npy", features) 
#     features = np.load("features.npy")


    return features





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
        # feats_1 = model(torch.tensor(image_data1))
        feats_1 = extract_feature(ARGS.image_one_path, model, device, tta = True)
        feats_1 = feats_1.cpu().numpy()

        # feats_2 = model(torch.tensor(image_data2))
        feats_2 = extract_feature(ARGS.image_two_path, model, device, tta = True)
        feats_2 = feats_2.cpu().numpy()
    dist = distance(feats_1, feats_2, ARGS.distance_metric)

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
    parser.add_argument('--distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

