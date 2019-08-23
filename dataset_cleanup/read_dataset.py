from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
person1_name
    label    0
    person1_subgroup_1
        file_path    '/path/to/file1'
        embedding    [4.5, 2.1, 9.9]
    person1_subgroup_2
        file_path    '/path/to/file123'
        embedding    [84.5, 32.32, 10.1]

person2_name
    label    1
    person2_subgroup_1
        file_path    '/path/to/file4444'
        embedding    [1.1, 2.1, 2.9]
    person2_subgroup_2
        file_path    '/path/to/file1123123'
        embedding    [3.0, 41.1, 56.621]




python3 dataset_cleanup/read_dataset.py

'''

import os
import h5py
import numpy as np

## Reading From File
# with h5py.File('data/dataset.h5', 'r') as f:
#     for person in f.keys():
#         print("personName: " + str(person))
#         print("personLabel: " + str(f[person].attrs['label']))

#         for subgroup in f[person].keys() :
#             print("\tsubgroup: " + str(subgroup))

#             print("\t\tembedding data: " + str(f[person][subgroup]['embedding'][:4]))
#             print("\t\tpath data: " + str(f[person][subgroup].attrs['file_path']))


import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

# Data for each person
with h5py.File('data/dataset.h5', 'r') as f:
    for person in f.keys():
        print("personName: " + str(person))
#         print("personLabel: " + str(f[person].attrs['label']))

        nrof_images = len(f[person].keys())
        embedding_size = 512
        embeddings_array = np.zeros((nrof_images, embedding_size))
        label_array = np.zeros((0,0))
        label_strings_array = []

        print("\tembedding array shape: " + str(embeddings_array.shape))
        print("\tnumber of images: " + str(nrof_images) + "  embedding size: " + str(embedding_size))

        for i, subgroup in enumerate(f[person].keys()):
            print("\tlabel: " + str(i))
            embeddings_array[i, :] = f[person][subgroup]['embedding'][:]
            label_array = np.append(label_array, i)
            label_strings_array.append(str(subgroup))
            
            print("\tsubgroup: " + str(subgroup))
            print("\t\tembedding data shape: " + str(f[person][subgroup]['embedding'][:].shape))

            print("\t\tembedding data: " + str(f[person][subgroup]['embedding'][:4]))
            print("\t\tpath data: " + str(f[person][subgroup].attrs['file_path']))

        plt.figure(figsize=(10, 7))
        plt.title(str(person))
        dend = shc.dendrogram(shc.linkage(embeddings_array, method='ward'),labels=label_strings_array,color_threshold=1.0)
        plt.show()