#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
# create subcloud labels and overlap region labels
# assign subcloud class labels to points
'''
import time
import numpy as np
import pickle
import torch
import math
# from partition.provider_reduce import read_spg, read_features
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors
from datasets.process_anchors import *

cloud_names =  ['Vaihingen3D_Train']

input_trees = []
input_colors = []
input_labels = []
tree_path = #

for i, f in enumerate(cloud_names):
    cloud_name = cloud_names[i]
    # Name of the input files
    KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
    sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
    # Check if inputs have already been computed
    if exists(KDTree_file):
        data = read_ply(sub_ply_file)
         #data['intensity'].T
        sub_labels = data['class']        
        # Read pkl with search tree
        with open(KDTree_file, 'rb') as f:
            search_tree = pickle.load(f)
    else:
        print(KDTree_file)
    input_trees += [search_tree]
    input_labels += [sub_labels]

all_data = dict()

sub_radius = 6
num_classes = 9
anchor_method = 'reduce1' #'full' 'reduce1'
for i, tree in enumerate(input_trees):
    print(i, tree.data.shape)
    points = np.array(tree.data)
    lbs = input_labels[i]
    anchor = get_anchors(points, sub_radius, method=anchor_method)
    print('lbs: ', np.unique(input_labels[i]))
    print('class_n: ', num_classes)
    anchor, anchor_tree, anchors_dict, achor_lb = anchors_part_lbs(tree, anchor, input_labels[i], sub_radius, num_classes)
    # print('original_anchor_num: ', )
    # update sub_region information according to averlaps
    anchor, anchor_tree, anchors_dict, achor_lb = update_anchors(tree, anchor, anchor_tree, anchors_dict, achor_lb, sub_radius, lbs)
 
    c_name = cloud_names[i]
    all_data[c_name] = anchor, anchor_tree, anchors_dict, achor_lb
    

all_class_lb = np.ones((points.shape[0], num_classes))
for aa in anchors_dict.keys():
    idx = anchors_dict[aa][0]
    lbs = achor_lb[aa]
    slc_lb = all_class_lb[idx]
    all_class_lb[idx] = slc_lb*slc_lb*lbs
    
class_lb = # 'class_lb.txt' save path
np.savetxt(class_lb, all_class_lb, fmt='%i', delimiter=' ')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    