from __future__ import print_function
import os
import cv2
import h5py
import numpy as np
from copy import deepcopy

#------------------------------------------------------------------------------
dataset_type     = 'test' # Change to train/test
dataset_dir      = './dataset'
curated_dataset  = os.path.join(dataset_dir, dataset_type + '_curated')
curated_textfile = os.path.join(dataset_dir, dataset_type + '.txt')
file_path        = './dataset/%s/' % (dataset_type)
mat_file         = './dataset/%s/digitStruct.mat' % (dataset_type)
fixed_num        = 20
img_size         = (64, 64) # (width, height)
max_steps        = 3
ground_attention_downsample = (7, 7)

#---------------------------- Functions ----------------------------------------
def gaussian2d(sup, scales):
    """
    Creates a 2D Gaussian based on the size and scale.
    """
    var   = scales * scales
    shape = (sup[0], sup[1])
    n,m   = [(i-1)/2 for i in shape]
    y,x   = np.ogrid[-m:m+1,-n:n+1]

    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var))

    return g

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)

def generate_ground_gaussian_attention_mask(sample, sample_top, sample_height, sample_left, sample_width):
    """
    Creates a ground truth attention mask based on ground truth bounding boxes,
    and scales to fit into the box.
    """
    sample_image_height, sample_image_width, _ = sample.shape
    # Convert even to odd by adding extra px
    if sample_width%2 == 0:
        sample_width += 1

    if sample_height%2 == 0:
        sample_height += 1

    # Create and normalize
    scales = np.sqrt(2) * 8 # Play with the standard deviation
    gaussain = gaussian2d((sample_width, sample_height), scales)
    gaussain_normalized = (gaussain - np.min(gaussain))/(np.max(gaussain) - np.min(gaussain))

    # Create and normalize
    sample_attention = np.zeros((sample_image_height, sample_image_width)) * 0.0
    sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = gaussain_normalized

    # Create and normalize
    sample_attention_res      = cv2.resize(sample_attention, ground_attention_downsample,\
                                       interpolation=cv2.INTER_NEAREST)
    sample_attention_res      = sample_attention_res.flatten()
    sample_attention_res_norm = softmax(sample_attention_res)
    sample_attention_res_norm = np.reshape(sample_attention_res_norm, ground_attention_downsample)

    return sample_attention, sample_attention_res_norm
#------------------------------------------------------------------------------
