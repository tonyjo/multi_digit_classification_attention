from __future__ import print_function
import os
import cv2
import h5py
import argparse
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, help="extra/train/val/test")
parser.add_argument("--dataset_dir",  type=str, default="./dataset")
parser.add_argument("--gen_val_data", type=str, default="True")
parser.add_argument("--max_steps",    type=int, default=6, help="max steps")
parser.add_argument("--img_height",   type=int, default=64, help="image height")
parser.add_argument("--img_width",    type=int, default=64, help="image width")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg,"'", ": ", getattr(args, arg))
print('----------------------------------------')

#----------------------------Arguments---------------------------------------
# Numpy seed values
np.random.seed(8964)

dataset_type     = args.dataset_type # Change to train/test
dataset_dir      = args.dataset_dir
gen_val_data     = args.gen_val_data
curated_dataset  = os.path.join(dataset_dir, dataset_type + '_cropped')
curated_textfile = os.path.join(dataset_dir, dataset_type + '.txt')
file_path        = './dataset/%s/' % (dataset_type)
mat_file         = './dataset/%s/digitStruct.mat' % (dataset_type)
expand_percent   = 30
img_size         = (args.img_width, args.img_height) # (width, height)
max_steps        = args.max_steps
total_data       = 0

if dataset_type == "train" and gen_val_data == "True":
    curated_val_textfile = os.path.join(dataset_dir, 'val.txt')
    vt = open(curated_val_textfile, 'w')
    total_val_data = 0
elif dataset_type == "train" and gen_val_data == "False":
    curated_dataset  = os.path.join(dataset_dir, dataset_type + '_noval_cropped')
    curated_textfile = os.path.join(dataset_dir, dataset_type + '_noval.txt')

if os.path.exists(curated_dataset) == False:
    os.mkdir(curated_dataset)

#---------------------------- Functions ----------------------------------------
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
        # Matlab to python --indexing correction
        if key == 'left'or key == 'top':
            values_idx_corr = [val-1 for val in values]
            all_iterm_data.append(values_idx_corr)
        else:
            all_iterm_data.append(values)

    return all_iterm_data

def biggest_box(all_data, sample_index, total_samples):
    all_left  = []
    all_top   = []
    all_width = []
    all_heigt = []

    for k in range(total_samples):
        sample_left  = abs(int(all_data[sample_index][1][1][k]))
        sample_top   = abs(int(all_data[sample_index][1][2][k]))
        sample_width = abs(int(all_data[sample_index][1][3][k]))
        sample_heigt = abs(int(all_data[sample_index][1][4][k]))

        all_left.append(sample_left)
        all_top.append(sample_top)
        all_width.append(sample_left+sample_width)
        all_heigt.append(sample_top+sample_heigt)

    low_left = min(all_left)
    low_top  = min(all_top)
    highest_width = max(all_width) - low_left
    highest_height = max(all_heigt) - low_top

    return low_left, low_top, highest_width, highest_height

def new_x(low_x, delta_W, width, image_width):
    # Lower bound
    if delta_W//2 >= low_x:
        new_left = 0 # Reset to zero
        change1w = low_x
    elif (low_x - delta_W//2) > 0:
        new_left = low_x - delta_W//2
        change1w = delta_W//2
    # Upper Bound
    if (low_x + width + delta_W//2) < image_width:
        change2w = delta_W//2
    else:
        change2w = image_width - (low_x + width) - 1 # Set to max-width
    # New width
    new_width = width + change1w + change2w

    return new_left, new_width

def new_y(low_y, delta_H, height, image_height):
    # Lower bound
    if (delta_H//2) >= low_y:
        new_top  = 0 # Reset to zero
        change1h = low_y
    elif (low_y - delta_H//2) > 0:
        new_top  = low_y - delta_H//2
        change1h = delta_H//2
    # Upper Bound
    if (low_y + height + delta_H//2) < image_height:
        change2h = delta_H//2
    else:
        change2h = image_height - (low_y + height) - 1 # Set to max-height

    # New height
    new_height = height + change1h + change2h

    return new_top, new_height
#-------------------------------------------------------------------------------


#--------------------------------- MAIN ----------------------------------------
expand_percent = expand_percent/100.0

f = h5py.File(mat_file,'r')
print('Total bboxes: ', f['/digitStruct/name'].shape[0])

all_data = []
for j in range(f['/digitStruct/bbox'].shape[0]):
#for j in range(1000):
    img_name = get_name(j, f)
    row_dict = get_bbox(j, f)
    all_data.append([img_name, row_dict])
    if j%4000 == 0:
        print('Completion..{%d/%d}' % (j, f['/digitStruct/bbox'].shape[0]))
print('Completion..{%d/%d}' % (f['/digitStruct/bbox'].shape[0], f['/digitStruct/bbox'].shape[0]))
print('Completed!')

# Create copy of samples to curate labels
all_data_copy = deepcopy(all_data)
#-------------------------------------------------------------------------------

with open(curated_textfile, 'w') as ft:
    for sample_index in range(len(all_data)):
        random_value = np.random.random()
        try:
            sample_imgph = all_data[sample_index][0]
            sample_image = cv2.imread(file_path+sample_imgph)
            sample_image_copy_ = deepcopy(sample_image)
            sample_heigt_org, sample_width_org, _ = sample_image.shape
            # Get how many digits:
            total_samples = np.array(all_data[sample_index][1]).shape[1]
            #-------------------------------------------------------------------
            ## Get bounding box encompassing all digits
            low_left, low_top, high_width, high_height = biggest_box(all_data=all_data,\
                                  sample_index=sample_index, total_samples=total_samples)
            #-------------------------------------------------------------------
            ## Crop
            # Obtain the spatial extend to crop
            high_width_expand  = np.floor(expand_percent * high_width)
            high_height_expand = np.floor(expand_percent * high_height)
            # Expand in x-direction
            lx, nW = new_x(low_x=low_left, delta_W=high_width_expand,\
                           width=high_width, image_width=sample_width_org)
            # Expand in y-direction
            ly, nH = new_y(low_y=low_top, delta_H=high_height_expand,\
                           height=high_height, image_height=sample_heigt_org)
            #------- X-axis shift -------
            for i in range(total_samples):
                # Update lower bound-- left
                all_data_copy[sample_index][1][1][i] -= lx
            #------- Y-axis shift-------
            for i in range(total_samples):
                # Update lower bound-- top
                all_data_copy[sample_index][1][2][i] -= ly
            # Crop image
            sample_image_copy = sample_image_copy_[int(ly):int(ly + nH), int(lx):int(lx + nW), :]
            #-------------------------------------------------------------------
            ## Resize image
            sample_heigt_org_rz, sample_width_org_rz, _ = sample_image_copy.shape

            if sample_width_org_rz > img_size[0]:
                # Shrinking
                smpl_img_rz = cv2.resize(sample_image_copy, img_size, interpolation = cv2.INTER_AREA)
            else:
                # Zooming
                smpl_img_rz = cv2.resize(sample_image_copy, img_size, interpolation = cv2.INTER_LINEAR)
            #-------------------------------------------------------------------
            ## Curate labels to match new image shape
            # Collect samples
            samples = []
            for index_into in range(max_steps):
                if index_into == 0:
                    sample_label = 0
                    sample_left  = 0
                    sample_top   = 0
                    sample_width = 1
                    sample_heigt = 1
                elif index_into > total_samples:
                    sample_label = 11
                    sample_left  = img_size[0] - 2
                    sample_top   = img_size[1] - 2
                    sample_width = 1
                    sample_heigt = 1
                else:
                    sample_label = int(all_data_copy[sample_index][1][0][index_into-1])
                    sample_left  = int(np.ceil((all_data_copy[sample_index][1][1][index_into-1]  * img_size[0])/sample_width_org_rz))
                    sample_top   = int(np.ceil((all_data_copy[sample_index][1][2][index_into-1]  * img_size[1])/sample_heigt_org_rz))
                    sample_width = int(np.floor((all_data_copy[sample_index][1][3][index_into-1] * img_size[0])/sample_width_org_rz))
                    sample_heigt = int(np.floor((all_data_copy[sample_index][1][4][index_into-1] * img_size[1])/sample_heigt_org_rz))
                # Append
                samples.append([all_data_copy[sample_index][0][:-4], sample_label, sample_left, sample_top, sample_width, sample_heigt])
            #-------------------------------------------------------------------
            ## Write and save
            # New sample Image path
            new_sample_image_path = os.path.join(curated_dataset, all_data_copy[sample_index][0])
            # Save
            cv2.imwrite(new_sample_image_path, smpl_img_rz)

            if dataset_type == "train" and gen_val_data == "True":
                if random_value < 0.1:
                    # Write to validation
                    vt.write(str(samples))
                    vt.write('\n')
                    total_val_data += 1
                else:
                    # Write to train
                    ft.write(str(samples))
                    ft.write('\n')
                    total_data += 1
            else:
                # Write
                ft.write(str(samples))
                ft.write('\n')
                total_data += 1
            # Free memory
            del sample_image, sample_image_copy, sample_image_copy_
            #-------------------------------------------------------------------
            if sample_index%4000 == 0:
                print('Completion..{%d/%d}' % (sample_index, len(all_data)))
        #-----------------------------------------------------------------------
        except ValueError:
            print('Ignoring data: ', sample_imgph)
#-------------------------------------------------------------------------------
print('Completion..{%d/%d}' % (len(all_data), len(all_data)))
print('Completed!')
# Close
ft.close()
if dataset_type == "train"  and gen_val_data == "True":
    vt.close()
    print('Total validation data = %d' % (total_val_data))
print('Total %s data = %d' % (dataset_type, total_data))
#------------------------------------------------------------------------------
