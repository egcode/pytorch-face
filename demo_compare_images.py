from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''

#################################################################################
#################################################################################
#################################################################################
ARCFACE LOSS - face.evoLVe.PyTorch
#################################################################################

##### NOT SAME        without extract_feature distance = [1.043365702033043] 
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Liuba/IMG_0179.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [1.0594086982309818]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Alex/haweF.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.9408659227192402]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Eugene/hawfd.png \
--image_two_path ./data/golovan_160/Julia/IMG_1922.png \
--distance_metric 1


##### NOT SAME        without extract_feature distance = [0.8103613704442978]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Eugene/IMG_0083.png \
--image_two_path ./data/golovan_160/Alex/hawefds.png \
--distance_metric 1





##### SAME            without extract_feature distance = [0.4351288080215454]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.9727675095200539]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0004.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.7395100891590118]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Eugene/IMG_9489.png \
--image_two_path ./data/golovan_160/Eugene/IMG_E7066.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.37082868814468384]
python3 demo_compare_images.py \
--model ./pth/backbone_ir50_ms1m_epoch120.pth \
--image_one_path ./data/golovan_160/Alex/harfds.png \
--image_two_path ./data/golovan_160/Alex/hawefd.png \
--distance_metric 1


#################################################################################
#################################################################################
#################################################################################
ARCFACE LOSS-Eugene Casia
#################################################################################


##### NOT SAME        without extract_feature distance = [1.0465557985007763] 
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Liuba/IMG_0179.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.708492249250412]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Alex/haweF.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.8642878979444504]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Eugene/hawfd.png \
--image_two_path ./data/golovan_160/Julia/IMG_1922.png \
--distance_metric 1


##### NOT SAME        without extract_feature distance = [0.8804192990064621]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Eugene/IMG_0083.png \
--image_two_path ./data/golovan_160/Alex/hawefds.png \
--distance_metric 1





##### SAME            without extract_feature distance = [0.1647181510925293]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.27410292625427246]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0004.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.163324773311615]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Eugene/IMG_9489.png \
--image_two_path ./data/golovan_160/Eugene/IMG_E7066.png \
--distance_metric 1

##### SAME            without extract_feature distance = [0.0513153076171875]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_arcface_casia_epoch21.pth \
--image_one_path ./data/golovan_160/Alex/harfds.png \
--image_two_path ./data/golovan_160/Alex/hawefd.png \
--distance_metric 1



#################################################################################
#################################################################################
#################################################################################
CENTER LOSS
#################################################################################



##### NOT SAME        without extract_feature distance = [1.1721499] 
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Liuba/IMG_0179.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 0

##### NOT SAME        without extract_feature distance = [1.2014594]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Alex/haweF.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 0

##### NOT SAME        without extract_feature distance = [1.2041703]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Eugene/hawfd.png \
--image_two_path ./data/golovan_160/Julia/IMG_1922.png \
--distance_metric 0


##### NOT SAME        without extract_feature distance = [0.86273426]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Eugene/IMG_0083.png \
--image_two_path ./data/golovan_160/Alex/hawefds.png \
--distance_metric 0





##### SAME            without extract_feature distance = [0.58154285]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0001.png \
--distance_metric 0

##### SAME            without extract_feature distance = [0.8795128]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Julia/0003.png \
--image_two_path ./data/golovan_160/Julia/0004.png \
--distance_metric 0

##### SAME            without extract_feature distance = [0.50008076]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Eugene/IMG_9489.png \
--image_two_path ./data/golovan_160/Eugene/IMG_E7066.png \
--distance_metric 0

##### SAME            without extract_feature distance = [0.25166833]
python3 demo_compare_images.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth \
--image_one_path ./data/golovan_160/Alex/harfds.png \
--image_two_path ./data/golovan_160/Alex/hawefd.png \
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

    ###### IMAGE 1111
    img_path1 = ARGS.image_one_path
    img1 = Image.open(img_path1)
    image_data1 = img1.convert('RGB')
    image_data_rgb_1 = np.asarray(image_data1) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    ccropped_1, flipped_1 = crop_and_flip(image_data_rgb_1, for_dataloader=False)
    # image_data1.save('pilllllllll.png')

    ###### IMAGE 2222
    img_path2 = ARGS.image_two_path
    img2 = Image.open(img_path2)
    image_data2 = img2.convert('RGB')
    image_data_rgb_2 = np.asarray(image_data2) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    ccropped_2, flipped_2 = crop_and_flip(image_data_rgb_2, for_dataloader=False)

    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()
    #########################################

    with torch.no_grad():
        feats_1 = extract_norm_features(ccropped_1, flipped_1, model, device, tta = True)
        feats_1 = feats_1.cpu().numpy()

        feats_2 = extract_norm_features(ccropped_2, flipped_2, model, device, tta = True)
        feats_2 = feats_2.cpu().numpy()

    dist = distance(feats_1, feats_2, ARGS.distance_metric)

    print("======DISTANCE===========")
    print(dist)
    print("=================")
    

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

