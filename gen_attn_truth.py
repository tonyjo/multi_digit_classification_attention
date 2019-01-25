from __future__ import print_function
import os
import cv2
import h5py
import numpy as np

#----------------------------Arguments---------------------------------------
dataset_type     = 'test' # Change to train/test
dataset_dir      = './dataset'
curated_dataset  = os.path.join(dataset_dir, dataset_type + '_cropped')
curated_textfile = os.path.join(dataset_dir, dataset_type + '.txt')
ground_attn_dir  = os.path.join(dataset_dir, dataset_type + '_attn_grnd')
file_path        = './dataset/%s/' % (dataset_type)
img_size         = (64, 64) # (width, height)
max_steps        = 7
ground_attention_downsample = (7, 7)

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
    total_samples= 0
    for k in range(max_steps):
        sample_label = samples[k][1]

        if int(sample_label) != 0:
            total_samples += 1

    return total_samples

def gaussian2d(image_height, image_width):
    """
    Creates a 2D Gaussian based on the size and scale.
    """
    x, y = np.meshgrid(np.linspace(-10,10,image_width),\
                       np.linspace(-10,10,image_height))

    d = np.sqrt(x*x+y*y)
    sigma, mu = 5.0, 0.0
    g = np.exp(-((d-mu)**2 / ( 2.0 * sigma**2 )))

    return g

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)

def generate_ground_gaussian_attention_mask(sample_top, sample_height, sample_left, sample_width):
    """
    Creates a ground truth attention mask based on ground truth bounding boxes,
    and scales to fit into the box.
    """
    sample_image_height, sample_image_width = img_size

    if sample_height > sample_image_height:
        sample_height = sample_image_height

    if sample_top + sample_height > sample_image_height:
        delta_y = (sample_top + sample_height) - sample_image_height
        sample_top = sample_top - delta_y

    if sample_width > sample_image_width:
        sample_width = sample_image_width

    if sample_left + sample_width > sample_image_width:
        delta_x = (sample_left + sample_width) - sample_image_width
        sample_left = sample_left - delta_x

    # Normalize each value between 0 and 1
    gaussain = gaussian2d(sample_height, sample_width)
    gy, gx = gaussain.shape
    # print(sample_height, sample_width)
    # print(gaussain.shape)
    # print(sample_top, sample_left)

    # Generate new array
    sample_attention = np.zeros((sample_image_height, sample_image_width)) * 0.0
    sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = gaussain
    # print(sample_attention.shape)

    # Downsample and re-normalize between 0 and 1
    sample_attention_res = cv2.resize(sample_attention, ground_attention_downsample, interpolation=cv2.INTER_NEAREST)
    sample_attention_res_norm = sample_attention_res/(np.sum(sample_attention_res) + 1e-5)
    sample_attention_res_norm = sample_attention_res_norm.flatten()

    # Sample
    max_idx  = np.argmax(sample_attention_res_norm)
    sort_idx = np.argsort(sample_attention_res_norm)
    # print('****************')
    sample_attention_res_norm_new = np.zeros((ground_attention_downsample)) * 0.0
    sample_attention_res_norm_new = sample_attention_res_norm_new.flatten()

    sample_attention_res_norm_new[max_idx] = 0.7

    if sort_idx[-2] == max_idx:
        sample_attention_res_norm_new[sort_idx[-3]] = 0.3
    else:
        sample_attention_res_norm_new[sort_idx[-2]] = 0.2
        sample_attention_res_norm_new[sort_idx[-3]] = 0.1

    return sample_attention, sample_attention_res_norm_new

def biggest_box(samples, total_samples):
    """
    Compute the box encompassing all the digits.
    """
    all_left  = []
    all_top   = []
    all_width = []
    all_heigt = []

    for k in range(total_samples):
        sample_label = samples[k][1]
        sample_left  = samples[k][2]
        sample_top   = samples[k][3]
        sample_width = samples[k][4]
        sample_heigt = samples[k][5]

        all_left.append(sample_left)
        all_top.append(sample_top)
        all_width.append(sample_left+sample_width)
        all_heigt.append(sample_top+sample_heigt)

    low_left = min(all_left)
    low_top  = min(all_top)
    highest_width = max(all_width) - low_left
    highest_height = max(all_heigt) - low_top

    return low_left, low_top, highest_width, highest_height

def generate_start_attention_mask_v1():
    """
    Create an attention mask for the stop state, which is a
    uniform mask around the digits.
    """
    sample_image_height, sample_image_width = img_size

    x = np.ones((sample_image_height, sample_image_width)) * 1.0
    x = x / (sample_image_height*sample_image_width)

    # Downsample and re-normalize between 0 and 1
    x1 = np.zeros(ground_attention_downsample) * 0.0
    x1[0, 0] = 0.25
    x1[0, 6] = 0.25
    x1[6, 0] = 0.25
    x1[6, 6] = 0.25

    return x, x1

def generate_stop_attention_mask_v1(samples, total_samples):
    """
    Create an attention mask for the stop state, which is a
    uniform mask around the digits.
    """
    sample_image_height, sample_image_width = img_size

    low_left, low_top, highest_width, highest_height = biggest_box(samples, total_samples)

    x = np.ones((sample_image_height, sample_image_width))

    mask = np.ones((sample_image_height, sample_image_width))
    mask[low_top:low_top+highest_height, low_left:low_left+highest_width] = 0

    x2 = np.multiply(x, mask)

    # Downsample and re-normalize between 0 and 1
    x3 = cv2.resize(x2, ground_attention_downsample, interpolation=cv2.INTER_NEAREST)
    x3 = np.float32(x3)

    x3_1 = x3/np.sum(x3 + 1e-9) + 1e-5 # For numerical stability

    return x2, x3_1

def generate_stop_attention_mask_v2():
    """
    Create an attention mask for the stop state, which is a
    uniform mask around the digits.
    """
    sample_image_height, sample_image_width = img_size

    x = np.zeros((sample_image_height, sample_image_width)) * 0.0
    x[-1, -1] = 1.0

    # Downsample and re-normalize between 0 and 1
    x1 = np.zeros(ground_attention_downsample) * 0.0
    x1[-1, -1] = 1.0

    return x, x1

#-------------------------------------------------------------------------------


#--------------------------------- MAIN ----------------------------------------
# Load data
all_data = load_file(curated_textfile)
print('Data loaded!')

for sample_index in range(len(all_data)):
#for sample_index in [988]:
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
            _, attn_mask = generate_start_attention_mask_v1()
        elif int(sample_label) == 11:
            # End state
            _, attn_mask = generate_stop_attention_mask_v2()
        else:
            # Digit attention mask
            _, attn_mask = generate_ground_gaussian_attention_mask(sample_top,\
                                        sample_heigt, sample_left, sample_width)

        # Save attention mask
        attn_save_path = os.path.join(ground_attn_dir, all_data[sample_index][0][0][:-4] + '_' + str(k) + '.npy')
        np.save(attn_save_path, attn_mask)

    if sample_index%4000 == 0:
        print('Completion..{%d/%d}' % (sample_index, len(all_data)))

print('Completion..{%d/%d}' % (len(all_data), len(all_data)))
print('Completed!')
#-------------------------------------------------------------------------------
