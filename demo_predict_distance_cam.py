
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''

python3 demo_predict_distance_cam.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch34.pth \
--embeddings_premade ./output_arrays/embeddings_center_1.npy \
--label_string_center ./output_arrays/label_strings_center_1.npy \
--labels_center ./output_arrays/labels_center_1.npy


python3 demo_predict_distance_cam.py \
--model ./pth/IR_50_MODEL_centerloss_casia_epoch34.pth \
--embeddings_premade ./output_arrays/embeddings_center_2.npy \
--label_string_center ./output_arrays/label_strings_center_2.npy \
--labels_center ./output_arrays/labels_center_2.npy \
--distance_metric 1

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


max_threshold = 1.0

unknown_class = "unknown"  # unknown folder

class Face:
    def __init__(self):
        self.name = None
        self.distance = None
        self.bounding_box = None
        self.image = None
        self.embedding = None
        self.all_results_dict = {}

    def parse_all_results_dict(self):
        average_dist_dict = {}
        for key, distances_arr in self.all_results_dict.items():
            average_dist_dict[key] = np.mean(distances_arr)

        name = min(average_dist_dict, key=average_dist_dict.get) #get minimal value from dictionary
        self.distance = average_dist_dict[name]

        if average_dist_dict[name] < max_threshold: 
            self.name = name
        else:
            self.name = unknown_class

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32, gpu_memory_fraction = 0.3):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image, image_size):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            # face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            # faces.append(face)

            # cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = prewhiten(aligned)        
            face = Face()
            face.image = prewhitened
            face.bounding_box = bb
            faces.append(face)

        return faces

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def main(ARGS):
  
    vs = VideoStream(src=0).start() # regular webcam camera
    # vs = VideoStream(usePiCamera=True).start() # raspberry pi camera 

    embeddings_premade = np.load(ARGS.embeddings_premade, allow_pickle=True)
    label_string_center = np.load(ARGS.label_string_center, allow_pickle=True)
    labels_center = np.load(ARGS.labels_center, allow_pickle=True)


    detect = Detection()
      
    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()


    while True:

        frame = vs.read()
        # ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=400)

        #########################################

        faces = detect.find_faces(frame, ARGS.image_size)
        for face in faces:
            face.distance = 9
        for i, face in enumerate(faces):
            # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            normalize = T.Normalize(mean=[0.5], std=[0.5])
            transforms = T.Compose([
                T.Resize([ARGS.image_size,ARGS.image_size]),
                T.ToTensor(),
                normalize
            ])
            pil_image = Image.fromarray(face.image.astype('uint8'), 'RGB')
            pil_image = transforms(pil_image)
            pil_image = pil_image.reshape(1, 3, ARGS.image_size, ARGS.image_size)

            with torch.no_grad():
                feats = model(torch.tensor(pil_image))
                face.embedding = feats.cpu().numpy()


            # feed_image = np.expand_dims(face.image, axis=0)
            # feed_dict = { images_placeholder: feed_image , phase_train_placeholder:False}
            # face.embedding = sess.run(embeddings, feed_dict=feed_dict)

        nrof_premade = embeddings_premade.shape[0]
        
        for i in range(len(faces)):
            for j in range(nrof_premade):
                face = faces[i]
                # dist = np.sqrt(np.sum(np.square(np.subtract(face.embedding, embeddings_premade[j,:]))))
                
                dist = distance(face.embedding, embeddings_premade[j,:].reshape((1, 512)), ARGS.distance_metric)
                print("Distance: {}".format(dist))

                label = label_string_center[j]
                if label in face.all_results_dict: # if label value in dictionary
                    arr = face.all_results_dict.get(label)
                    arr.append(dist)
                else:
                    face.all_results_dict[label] = [dist]
                # print("candidate: " + str(i) + " distance: " + str(dist) + " with " + label_string_center[j])
        
        for i in range(len(faces)):
            # print("FACE :" + str(i))
            # print(faces[i].all_results_dict)
            faces[i].parse_all_results_dict()

        add_overlays(frame, faces)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

                    
                   
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def add_overlays(frame, faces):
    color_positive = (0, 255, 0)
    color_negative = (0, 0, 255)
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)

            color = color_negative
            name = unknown_class
            if face.distance is not None:
                if face.distance < max_threshold:
                    if face.name != unknown_class:
                        color = color_positive
                        name = face.name

            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          color, 2)

            if face.name == unknown_class:
                    cv2.putText(frame, name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                            thickness=2, lineType=2)
            elif face.name is not None and face.name:
                    cv2.putText(frame, name + " " + str(round(face.distance, 2)), (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                            thickness=2, lineType=2)

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
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--embeddings_premade', type=str, help='Premade embeddings array .npy format')
    parser.add_argument('--label_string_center', type=str, help='Premade label strings array .npy format')
    parser.add_argument('--labels_center', type=str, help='Premade labels integers array .npy format')
    parser.add_argument('--distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

