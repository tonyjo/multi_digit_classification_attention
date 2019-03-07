from __future__ import print_function
import os
import cv2
import h5py
import numpy as np
import argparse

# Numpy seed values
np.seed(8964)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, help="train/test")
args = parser.parse_args()
#----------------------------Arguments---------------------------------------
dataset_type     = args.dataset_type # Change to train/test
dataset_dir      = './dataset'
curated_dataset  = os.path.join(dataset_dir, dataset_type + '_cropped')
curated_textfile = os.path.join(dataset_dir, dataset_type + '.txt')
ground_attn_dir  = os.path.join(dataset_dir, dataset_type + '_attn_grnd')
file_path        = './dataset/%s/' % (dataset_type)
img_size         = (64, 64) # (width, height)
max_steps        = 6
ground_attention_size = (14, 14) # (width, height)

if os.path.exists(ground_attn_dir) == False:
    os.mkdir(ground_attn_dir)

#---------------------------- Functions ----------------------------------------
def load_file(curated_textfile):
    all_data = []

    with open(curated_textfile, 'r') as f:
        frames = f.readlines()

    for frame in frames:
        frame = frame.split(', ')
        iterm_data = []

        # Remove all non-interger characters
        for i in frame:
            i = i.replace("[", "")
            i = i.replace("[[", "")
            i = i.replace("]", "")
            i = i.replace("]]", "")
            i = i.replace("'", "")

            iterm_data.append(int(i))

        final_data = []

        count = 0
        for u in range(max_steps):
            each_data = []
            for k in range(6):
                if k == 0:
                    each_data.append(str(iterm_data[count]) + '.png')
                else:
                    each_data.append(iterm_data[count])

                count += 1

            final_data.append(each_data)

        all_data.append(final_data)

    return all_data

def samples_total(samples):
    """
    Computes the total digits in a sample
    """
    total_samples= 0
    for k in range(max_steps):
        sample_label = samples[k][1]

        if int(sample_label) != 0:
            total_samples += 1

    return total_samples

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

def generate_ground_gaussian_attention_mask(sample_size, sample_top, sample_height, sample_left, sample_width):
    """
    Creates a ground truth attention mask based on ground truth bounding boxes,
    and scales to fit into the box.
    """
    sample_image_height, sample_image_width = sample_size
    scales = np.sqrt(2) * 5 # Play with the standard deviation

    # Check size:
    if sample_left + sample_width >= sample_image_width:
        delta_w = (sample_left + sample_width) - sample_image_width
        sample_width -= delta_w

    if sample_top + sample_height >= sample_image_height:
        delta_h = (sample_top + sample_height) - sample_image_height
        sample_height -= delta_h

    # Convert even to odd by adding or removing extra px
    if sample_width%2 == 0:
        if sample_left + sample_width + 1 >= sample_image_width:
            sample_width -= 1
        else:
            sample_width += 1

    if sample_height%2 == 0:
        if sample_top + sample_height + 1 >= sample_image_height:
            sample_height -= 1
        else:
            sample_height += 1

    # Generate gaussian
    gaussain = gaussian2d((sample_width, sample_height), scales)
    gaussain_normalized = (gaussain - np.min(gaussain))/\
                          (np.max(gaussain) - np.min(gaussain))
    h_gauss_norm, w_gauss_norm = gaussain_normalized.shape
    gaussain_normalized = gaussain_normalized.flatten()
    gaussain_normalized = softmax(gaussain_normalized)
    gaussain_normalized = np.reshape(gaussain_normalized, (h_gauss_norm, w_gauss_norm))

    sample_attention = np.zeros((sample_image_height, sample_image_width)) * 0.0

    sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = gaussain_normalized

    return sample_attention

def generate_start_attention_mask(attn_size):
    """
    Create an attention mask for the start state.
    """
    x = np.zeros(attn_size) * 0.0
    x[0, 0] = 1.0

    return x

def generate_stop_attention_mask(attn_size):
    """
    Create an attention mask for the stop state.
    """
    x = np.zeros(attn_size) * 0.0
    x[-1, -1] = 1.0

    return x
#-------------------------------------------------------------------------------


#--------------------------------- MAIN ----------------------------------------
# Load data
all_data = load_file(curated_textfile)
print('Data loaded!')

for sample_index in range(len(all_data)):
#for sample_index in range(1000):
    #print(sample_index)
    sample_image_path = curated_dataset + '/' + all_data[sample_index][0][0]

    for k in range(max_steps):
        sample_left  = abs(int(all_data[sample_index][k][2]))
        sample_top   = abs(int(all_data[sample_index][k][3]))
        sample_width = abs(int(all_data[sample_index][k][4]))
        sample_heigt = abs(int(all_data[sample_index][k][5]))
        sample_label = all_data[sample_index][k][1]

        # Generate attention mask
        if k == 0:
            # Start state
            attn_mask = generate_start_attention_mask(attn_size=ground_attention_size)
        elif int(sample_label) == 11:
            # End state
            attn_mask = generate_stop_attention_mask(attn_size=ground_attention_size)
        else:
            # Digit attention mask
            # Rescale axis
            sample_left  = int(np.floor((sample_left  * ground_attention_size[0])/img_size[0]))
            sample_top   = int(np.floor((sample_top   * ground_attention_size[1])/img_size[1]))
            sample_width = int(np.floor((sample_width * ground_attention_size[0])/img_size[0]))
            sample_heigt = int(np.floor((sample_heigt * ground_attention_size[1])/img_size[1]))

            attn_mask = generate_ground_gaussian_attention_mask(ground_attention_size,\
                                       sample_top, sample_heigt, sample_left, sample_width)

        # Save attention mask
        attn_save_path = os.path.join(ground_attn_dir,\
                      all_data[sample_index][0][0][:-4] + '_' + str(k) + '.npy')
        np.save(attn_save_path, attn_mask)

    if sample_index%4000 == 0:
        print('Completion..{%d/%d}' % (sample_index, len(all_data)))

print('Completion..{%d/%d}' % (len(all_data), len(all_data)))
print('Completed!')
#-------------------------------------------------------------------------------
