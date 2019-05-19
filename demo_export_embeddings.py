
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

"""
Exports the embeddings and labels of a directory of images as numpy arrays.

Typicall usage expect the image directory to be of the openface/facenet form and
the images to be aligned. Simply point to your model and your image directory:

Output:
embeddings.npy -- Embeddings as np array, Use --embeddings_name to change name
labels.npy -- Integer labels as np array, Use --labels_name to change name
label_strings.npy -- Strings from folders names, --labels_strings_name to change name


Use --image_batch to dictacte how many images to load in memory at a time.

If your images aren't already pre-aligned, use --is_aligned False

I started with compare.py from David Sandberg, and modified it to export
the embeddings. The image loading is done use the facenet library if the image
is pre-aligned. If the image isn't pre-aligned, I use the compare.py function.
I've found working with the embeddings useful for classifications models.

Charles Jekel 2017

python3 demo_export_embeddings.py ./pth/IR_50_MODEL_centerloss_casia_epoch34.pth ./data/golovan_160/ \
--is_aligned True \
--image_size 112 \
--embeddings_name ./output_arrays/embeddings_center_1.npy \
--labels_name ./output_arrays/labels_center_1.npy \
--labels_strings_name ./output_arrays/label_strings_center_1.npy
"""

class FacesDataset(data.Dataset):
    def __init__(self, image_list, label_list, names_list, num_classes, input_size, is_aligned):
        self.image_list = image_list
        self.label_list = label_list
        self.names_list = names_list
        self.num_classes = num_classes
        self.is_aligned = is_aligned
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        self.transforms = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')
        data = self.transforms(data)
        label = self.label_list[index]
        name = self.names_list[index]
        return data.float(), label, name

    def __len__(self):
        return len(self.image_list)

def main(ARGS):
    
    train_set = get_dataset(ARGS.data_dir)
    image_list, label_list, names_list = get_image_paths_and_labels(train_set)
    faces_dataset = FacesDataset(image_list, label_list, names_list, len(train_set), ARGS.image_size, ARGS.is_aligned)
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
    nrof_images = len(loader.dataset)

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    nam_array = np.chararray((nrof_images,))
    with torch.no_grad():
        for i, (data, label, name) in enumerate(loader):

            data, label = data.to(device), label.to(device)

            feats = model(data)
            emb = feats.cpu().numpy()
            lab = label.detach().cpu().numpy()

            nam_array[lab] = name
            lab_array[lab] = lab
            emb_array[lab, :] = emb

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
    np.save(ARGS.labels_strings_name, nam_array)


##########################################
    # Get input and output tensors
    # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    # nrof_images = len(image_list)
    # print('Number of images: ', nrof_images)
    # batch_size = ARGS.image_batch
    # if nrof_images % batch_size == 0:
    #     nrof_batches = nrof_images // batch_size
    # else:
    #     nrof_batches = (nrof_images // batch_size) + 1
    # print('Number of batches: ', nrof_batches)
    # # embedding_size = embeddings.get_shape()[1]
    # embedding_size = 512
    # emb_array = np.zeros((nrof_images, embedding_size))
    # start_time = time.time()

    # for i in range(nrof_batches):
    #     if i == nrof_batches -1:
    #         n = nrof_images
    #     else:
    #         n = i*batch_size + batch_size
    #     # Get images for the batch
    #     if ARGS.is_aligned is True:
    #         images = load_data(image_list[i*batch_size:n], False, False, ARGS.image_size)
    #     else:
    #         images = load_and_align_data(image_list[i*batch_size:n], ARGS.image_size, ARGS.margin, ARGS.gpu_memory_fraction)
    #     feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    #     # Use the facenet model to calcualte embeddings
    #     embed = sess.run(embeddings, feed_dict=feed_dict)
    #     emb_array[i*batch_size:n, :] = embed
    #     print('Completed batch', i+1, 'of', nrof_batches)

    # run_time = time.time() - start_time
    # print('Run time: ', run_time)

    # #   export emedings and labels
    # label_list  = np.array(label_list)

    # np.save(ARGS.embeddings_name, emb_array)
    # np.save(ARGS.labels_name, label_list)
    # label_strings = np.array(label_strings)
    # np.save(ARGS.labels_strings_name, label_strings[label_list])


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        print(image_paths[i])
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # bp()
        # if (len(bounding_boxes) == 0):
        #     print("++++++Zero")
        #     # bp()
        #     break;
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='pth model file')
    parser.add_argument('data_dir', type=str, help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    parser.add_argument('--is_aligned', type=str, help='Is the data directory already aligned and cropped?', default=True)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--image_batch', type=int, help='Number of images stored in memory at a time. Default 500.', default=500)
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    #   numpy file Names
    parser.add_argument('--embeddings_name', type=str, help='Enter string of which the embeddings numpy array is saved as.', default='embeddings.npy')
    parser.add_argument('--labels_name', type=str, help='Enter string of which the labels numpy array is saved as.', default='labels.npy')
    parser.add_argument('--labels_strings_name', type=str, help='Enter string of which the labels as strings numpy array is saved as.', default='label_strings.npy')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
