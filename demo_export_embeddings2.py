
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

from models.resnet import *
from models.irse import *

from helpers import *

"""

## ALL FAMILY
python3 demo_export_embeddings2.py ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth ./data/golovan_160/ \
--is_aligned True \
--image_size 112 \
--image_batch 5 \
--embeddings_name ./output_arrays/embeddings_center_1.npy \
--labels_name ./output_arrays/labels_center_1.npy \
--labels_strings_name ./output_arrays/label_strings_center_1.npy

## SHORT
python3 demo_export_embeddings2.py ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth ./data/golovan_demo/ \
--is_aligned True \
--image_size 112 \
--image_batch 5 \
--embeddings_name ./output_arrays/embeddings_center_1.npy \
--labels_name ./output_arrays/labels_center_1.npy \
--labels_strings_name ./output_arrays/label_strings_center_1.npy


## SHORT NOT ALIGNED
python3 demo_export_embeddings2.py ./pth/IR_50_MODEL_centerloss_casia_epoch16.pth ./data/golovan_demo_not_aligned/ \
--is_aligned False \
--image_size 112 \
--image_batch 5 \
--embeddings_name ./output_arrays/embeddings_center_1.npy \
--labels_name ./output_arrays/labels_center_1.npy \
--labels_strings_name ./output_arrays/label_strings_center_1.npy
"""

class FacesDataset(data.Dataset):
    def __init__(self, image_list, label_list, names_list, num_classes, is_aligned, image_size, margin, gpu_memory_fraction):
        self.image_list = image_list
        self.label_list = label_list
        self.names_list = names_list
        self.num_classes = num_classes

        self.is_aligned = is_aligned

        self.image_size = image_size
        self.margin = margin
        self.gpu_memory_fraction = gpu_memory_fraction

        self.static = 0

        # normalize = T.Normalize(mean=[0.5], std=[0.5])
        # self.transforms = T.Compose([
        #     T.Resize(image_size),
        #     T.ToTensor(),
        #     normalize
        # ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')

        print('\n✅✅✅ self.is_aligned: {}'.format(self.is_aligned))

        if self.is_aligned is 'True':
            print('####### Images already ALIGNED')
            image_data_rgb = np.asarray(data) # (160, 160, 3)
        else:
            print('####### Images are NOT ALIGNED')
            image_data_rgb = load_and_align_data(img_path, self.image_size, self.margin, self.gpu_memory_fraction)


        ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=True)
        # bp()
        # print("\n\n")
        # print("### image_data_rgb shape: " + str(image_data_rgb.shape))
        # print("### CCROPPED shape: " + str(ccropped.shape))
        # print("### FLIPPED shape: " + str(flipped.shape))
        # print("\n\n")
        
        ################################################
        ### SAVE
        prefix = 'notaligned_' + str(self.static) + str(self.names_list[index]) 
        # data.save(prefix + 'aaaaaa.png') # Save PIL
        image_BGR = cv2.cvtColor(image_data_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(prefix + '.png', image_BGR)
        self.static += 1
        ################################################


        
        # data = self.transforms(data)
        label = self.label_list[index]
        name = self.names_list[index]
        return ccropped, flipped, label, name

    def __len__(self):
        return len(self.image_list)

def main(ARGS):
    
    np.set_printoptions(threshold=sys.maxsize)

    out_dir = 'output_arrays/'
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    train_set = get_dataset(ARGS.data_dir)
    image_list, label_list, names_list = get_image_paths_and_labels(train_set)
    faces_dataset = FacesDataset(image_list=image_list, 
                                    label_list=label_list, 
                                    names_list=names_list, 
                                    num_classes=len(train_set), 
                                    is_aligned=ARGS.is_aligned, 
                                    image_size=ARGS.image_size, 
                                    margin=ARGS.margin, 
                                    gpu_memory_fraction=ARGS.gpu_memory_fraction)
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

    # ###### IMAGE
    # img_path = './data/test_image.png'
    # img = Image.open(img_path)
    # image_data = img.convert('RGB')
    # image_data_rgb = np.asarray(image_data) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    # ccropped_im, flipped_im = crop_and_flip(image_data_rgb, for_dataloader=False)
    # feats_im = extract_norm_features(ccropped_im, flipped_im, model, device, tta = True)

    
########################################
    # nrof_images = len(loader.dataset)
    nrof_images = len(image_list)

    emb_array = np.zeros((nrof_images, embedding_size))
    # lab_array = np.zeros((nrof_images,))
    lab_array = np.zeros((0,0))

    # nam_array = np.chararray((nrof_images,))
    batch_ind = 0
    with torch.no_grad():
        for i, (ccropped, flipped, label, name) in enumerate(loader):

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
            
            lab_array = np.append(lab_array,lab)
            
            # print("\n")
            # for j in range(len(ccropped)):
            #     dist = distance(feats_im.cpu().numpy(), np.expand_dims(emb_array[j+batch_ind], axis=0))
            #     # dist = distance(feats_im, feats[j])
            #     print("22222 Distance Eugene with {}  is  {}:".format(name[j], dist))
            # print("\n")


            batch_ind += len(ccropped)

            if i % 10 == 9:
                print('.', end='')
                sys.stdout.flush()
        print('')

    # embeddings = emb_array
    # np.save('embeddings.npy', embeddings) 
    # embeddings = np.load('lfw/embeddings.npy')

    run_time = time.time() - start_time
    print('Run time: ', run_time)

    #   export emedings and labels
    np.save(ARGS.embeddings_name, emb_array)
    np.save(ARGS.labels_name, lab_array)


    label_strings = np.array(label_strings)
    np.save(ARGS.labels_strings_name, label_strings[label_list])

    # bp()


    # embeddings = np.load('output_arrays/embeddings_center_1.npy')
    # labels = np.load('output_arrays/labels_center_1.npy')
    # strings = np.load('output_arrays/label_strings_center_1.npy')


def load_and_align_data(image_path, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('🎃  Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    print(image_path)
    img = misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    # prewhitened = prewhiten(aligned)
    # img = prewhitened

    img = aligned
    
    return img


# def prewhiten(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
#     y = np.multiply(np.subtract(x, mean), 1/std_adj)
#     return y  

# def to_rgb(img):
#     w, h = img.shape
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
#     return ret

# def crop(image, random_crop, image_size):
#     if image.shape[1]>image_size:
#         sz1 = int(image.shape[1]//2)
#         sz2 = int(image_size//2)
#         if random_crop:
#             diff = sz1-sz2
#             (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
#         else:
#             (h, v) = (0,0)
#         image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
#     return image
  
# def flip(image, random_flip):
#     if random_flip and np.random.choice([True, False]):
#         image = np.fliplr(image)
#     return image

# def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
#     nrof_samples = len(image_paths)
#     images = np.zeros((nrof_samples, image_size, image_size, 3))
#     for i in range(nrof_samples):
#         img = misc.imread(image_paths[i])
#         if img.ndim == 2:
#             img = to_rgb(img)
#         if do_prewhiten:
#             img = prewhiten(img)
#         img = crop(img, do_random_crop, image_size)
#         img = flip(img, do_random_flip)
#         images[i,:,:,:] = img
#     return images

# def distance(embeddings1, embeddings2, distance_metric=0):
#     if distance_metric==0:
#         # Euclidian distance
#         diff = np.subtract(embeddings1, embeddings2)
#         dist = np.sum(np.square(diff),1)
#     elif distance_metric==1:
#         # Distance based on cosine similarity
#         dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
#         norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
#         similarity = dot / norm
#         dist = np.arccos(similarity) / math.pi
#     else:
#         raise 'Undefined distance metric %d' % distance_metric 
        
#     return dist

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='pth model file')
    parser.add_argument('data_dir', type=str, help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    parser.add_argument('--is_aligned', type=str, help='Is the data directory already aligned and cropped?', default=True)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--image_batch', type=int, help='Number of images stored in memory at a time. Default 64.', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    #   numpy file Names
    parser.add_argument('--embeddings_name', type=str, help='Enter string of which the embeddings numpy array is saved as.', default='embeddings.npy')
    parser.add_argument('--labels_name', type=str, help='Enter string of which the labels numpy array is saved as.', default='labels.npy')
    parser.add_argument('--labels_strings_name', type=str, help='Enter string of which the labels as strings numpy array is saved as.', default='label_strings.npy')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
