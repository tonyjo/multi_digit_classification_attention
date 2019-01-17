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

def gaussian2d(sup, scales):
    """
    Creates a 2D Gaussian based on the size and scale.
    """
    var   = scales * scales
    shape = (sup[0], sup[1])
    n,m   = [(i-1)/2 for i in shape]
    x,y   = np.ogrid[-m:m+1,-n:n+1]
    incr_x = 1
    incr_y = 1
    # Getting the scale correctly
    while x.shape[0] != sup[1] or y.shape[1] != sup[0]:
        if x.shape[0] > sup[1]:
            incr_x -= 1
        elif x.shape[0] == sup[1]:
            incr_x = incr_x
        else:
            incr_x += 1

        if y.shape[1] > sup[0]:
            incr_y -= 1
        elif y.shape[1] == sup[0]:
            incr_y = incr_y
        else:
            incr_y += 1

        x,y = np.ogrid[-m:m+incr_x,-n:n+incr_y]

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
    scales = np.sqrt(2) * 8
    gaussain = gaussian2d((sample_width, sample_heigt), scales)
    gaussain_normalized = (gaussain - np.min(gaussain))/\
                          (np.max(gaussain) - np.min(gaussain))

    sample_attention = np.zeros_like(sample) * 0.0
    sample_attention = sample_attention[:, :, 0]

    sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = gaussain_normalized

    sample_attention_res  = cv2.resize(sample_attention, ground_attention_downsample, interpolation=cv2.INTER_NEAREST)

    sample_attention_res  = sample_attention_res.flatten()

    sample_attention_res_norm = softmax(sample_attention_res)

    return sample_attention, sample_attention_res_norm

def generate_ground_attention_mask(sample, sample_top, sample_height, sample_left, sample_width):
    """
    Creates a ground truth attention mask based on ground truth bounding boxes,
    and scales to fit into the box.
    """
    true_attn_mask = np.ones((sample_heigt, sample_width)) * 1.0

    sample_attention = np.zeros_like(sample) * 0.0
    sample_attention = sample_attention[:, :, 0]

    sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = true_attn_mask

    sample_attention_res = cv2.resize(sample_attention, ground_attention_downsample, interpolation=cv2.INTER_NEAREST)

    sample_attention_res = np.expand_dims(sample_attention_res, axis=-1)

    return sample_attention, sample_attention_res

#------------------------------------------------------------------------------
