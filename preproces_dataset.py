from __future__ import print_function
import os
import cv2
import h5py
import numpy as np
from copy import deepcopy

#------------------------------------------------------------------------------
dataset_type     = 'train' # Change to train/test
dataset_dir      = './dataset'
curated_dataset  = os.path.join(dataset_dir, dataset_type + '_curated')
curated_textfile = os.path.join(dataset_dir, dataset_type + '.txt')
file_path        = './dataset/%s/' % (dataset_type)
mat_file         = './dataset/%s/digitStruct.mat' % (dataset_type)
fixed_num        = 20
img_size         = (64, 64) # (width, height)
scales           = np.sqrt(2) * 8
max_steps        = 3
ground_attention_downsample = (7, 7)

#------------------------------------------------------------------------------
if os.path.exists(curated_dataset) == False:
    os.mkdir(curated_dataset)
#------------------------------------------------------------------------------

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    all_iterm_data = []
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        all_iterm_data.append(values)

    return all_iterm_data

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

    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )

    return g

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

    sample_attention_res = cv2.resize(sample_attention, ground_attention_downsample, interpolation=cv2.INTER_NEAREST)

    sample_attention_res_norm = (sample_attention_res - np.min(sample_attention_res))/\
                                (np.max(sample_attention_res) - np.min(sample_attention_res))

    sample_attention_res_norm = np.expand_dims(sample_attention_res_norm, axis=3)

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

f = h5py.File(mat_file,'r')
print('Total bboxes: ', f['/digitStruct/name'].shape[0])

all_data = []
for j in range(f['/digitStruct/bbox'].shape[0]):
#for j in range(test_range):
    img_name = get_name(j, f)
    row_dict = get_bbox(j, f)

    all_data.append([img_name, row_dict])

    if j%4000 == 0:
        print('Completion..{%d/%d}' % (j, f['/digitStruct/bbox'].shape[0]))

print('Completion..{%d/%d}' % (f['/digitStruct/bbox'].shape[0], f['/digitStruct/bbox'].shape[0]))
print('Completed!')

# Create copy of samples
all_data_copy = deepcopy(all_data)

#------------------------------------------------------------------------------
with open(curated_textfile, 'w') as ft:
    #for sample_index in range(test_range):
    for sample_index in range(len(all_data)):
        try:
            sample_imgph = all_data[sample_index][0]
            sample_image = cv2.imread(file_path+sample_imgph)
            sample_image_copy = sample_image

            sample_heigt_org, sample_width_org, _ = sample_image.shape
            #-------------------------------------------------------------------
            ## Bounds
            lower_x = 1000
            upper_x = 0
            lower_y = 1000
            upper_y = 0

            # check how many samples:
            total_samples = np.array(all_data[sample_index][1]).shape[1]

            for i in range(total_samples):
                sample_left  = abs(int(all_data[sample_index][1][1][i]))
                sample_top   = abs(int(all_data[sample_index][1][2][i]))
                sample_width = abs(int(all_data[sample_index][1][3][i]))
                sample_heigt = abs(int(all_data[sample_index][1][4][i]))

                if sample_left < lower_x:
                    lower_x = sample_left

                if sample_left + sample_width > upper_x:
                    upper_x = sample_left + sample_width

                if sample_top < lower_y:
                    lower_y = sample_top

                if sample_top + sample_heigt > lower_y:
                    upper_y = sample_top + sample_heigt

            #-------------------------------------------------------------------
            # Crop
            if lower_x - fixed_num > 0:
                start = abs(lower_x - fixed_num)
                sample_image_copy = sample_image_copy[:, start:, :]
                # Include fixed pixels from all lower bound-- left
                for i in range(total_samples):
                    all_data_copy[sample_index][1][1][i] -= start
                upper_x_copy = upper_x - start
            else:
                upper_x_copy = upper_x

            if upper_x + fixed_num < sample_width_org:
                sample_image_copy = sample_image_copy[:, :upper_x_copy + fixed_num, :]

            if lower_y - fixed_num > 0:
                end = abs(lower_y - fixed_num)
                sample_image_copy = sample_image_copy[end:, :, :]
                # Include fixed pixels from all upper bound-- top
                for i in range(total_samples):
                    all_data_copy[sample_index][1][2][i] -= end
                upper_y_copy = upper_y - end
            else:
                upper_y_copy = upper_y

            if upper_y + fixed_num > sample_heigt_org:
                sample_image_copy = sample_image_copy[:upper_y_copy + fixed_num, :, :]

            #-------------------------------------------------------------------
            # Get samples
            sample_heigt_org_rz, sample_width_org_rz, _ = sample_image_copy.shape
            smpl_img_rz = cv2.resize(sample_image_copy, img_size)

            if total_samples >=  max_steps:
                total_samples = max_steps
                # Collect samples
                samples = []
                samples_attention = []
                samples_attention_rz = []
                for index_into in range(total_samples):
                    sample_label = int( all_data_copy[sample_index][1][0][index_into])
                    sample_left  = abs(int((all_data_copy[sample_index][1][1][index_into] * img_size[0])/sample_width_org_rz))
                    sample_top   = abs(int((all_data_copy[sample_index][1][2][index_into] * img_size[1])/sample_heigt_org_rz))
                    sample_width = abs(int((all_data_copy[sample_index][1][3][index_into] * img_size[0])/sample_width_org_rz))
                    sample_heigt = abs(int((all_data_copy[sample_index][1][4][index_into] * img_size[1])/sample_heigt_org_rz))
                    # Generate attention ground truth masks
                    sample_attention, sample_attention_res = generate_ground_attention_mask(smpl_img_rz, sample_top, sample_heigt, sample_left, sample_width)
                    # Append
                    samples_attention.append([sample_attention])
                    samples_attention_rz.append([sample_attention_res])
                    samples.append([all_data_copy[sample_index][0][:-4], sample_label, sample_left, sample_top, sample_width, sample_heigt])
            else:
                # Collect samples
                prev_idx = 0
                samples  = []
                samples_attention = []
                samples_attention_rz = []
                for index_to in range(max_steps):
                    # Duplicate samples
                    if index_to + 1 <= total_samples:
                        prev_idx = index_to

                    sample_label = int( all_data_copy[sample_index][1][0][prev_idx])
                    sample_left  = abs(int((all_data_copy[sample_index][1][1][prev_idx] * img_size[0])/sample_width_org_rz))
                    sample_top   = abs(int((all_data_copy[sample_index][1][2][prev_idx] * img_size[1])/sample_heigt_org_rz))
                    sample_width = abs(int((all_data_copy[sample_index][1][3][prev_idx] * img_size[0])/sample_width_org_rz))
                    sample_heigt = abs(int((all_data_copy[sample_index][1][4][prev_idx] * img_size[1])/sample_heigt_org_rz))
                    # Generate attention ground truth masks
                    sample_attention, sample_attention_res = generate_ground_attention_mask(smpl_img_rz, sample_top, sample_heigt, sample_left, sample_width)
                    # Append
                    samples_attention.append([sample_attention])
                    samples_attention_rz.append([sample_attention_res])
                    samples.append([all_data_copy[sample_index][0][:-4], sample_label, sample_left, sample_top, sample_width, sample_heigt])

            #-------------------------------------------------------------------
            # New sample Image path
            new_sample_image_path = os.path.join(curated_dataset, all_data_copy[sample_index][0])
            # Save
            cv2.imwrite(new_sample_image_path, smpl_img_rz)
            for j in range(max_steps):
                # Save ground attention-- from path remove png and add npy
                sample_attn_image_path = os.path.join(dataset_dir, dataset_type + '_curated', all_data_copy[sample_index][0][:-4] + '_' + str(j) + '.npy')
                np.save(sample_attn_image_path, samples_attention_rz[j][0])
            # Write
            ft.write(str(samples))
            ft.write('\n')

            #-------------------------------------------------------------------
            if sample_index%4000 == 0:
                print('Completion..{%d/%d}' % (sample_index, len(all_data)))

        except ValueError:
            print('Ignoring data: ', sample_imgph)

# Close
ft.close()
print('Completion..{%d/%d}' % (len(all_data), len(all_data)))
print('Completed!')
#------------------------------------------------------------------------------
