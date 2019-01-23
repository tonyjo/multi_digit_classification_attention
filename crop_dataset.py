from __future__ import print_function
import os
import cv2
import h5py
import argparse
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, help="train/test")
parser.add_argument("--dataset_dir",  type=str, default="./dataset")
parser.add_argument("--max_steps",    type=int, default=5, help="max steps")
parser.add_argument("--img_height",   type=int, default=64, help="image height")
parser.add_argument("--img_width",    type=int, default=64, help="image width")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg,"'", ": ", getattr(args, arg))
print('----------------------------------------')

#----------------------------Arguments---------------------------------------
dataset_type     = args.dataset_type # Change to train/test
dataset_dir      = args.dataset_dir
curated_dataset  = os.path.join(dataset_dir, dataset_type + '_cropped')
curated_textfile = os.path.join(dataset_dir, dataset_type + '.txt')
file_path        = './dataset/%s/' % (dataset_type)
mat_file         = './dataset/%s/digitStruct.mat' % (dataset_type)
expand_percent   = 30
img_size         = (args.img_width, args.img_height) # (width, height)
max_steps        = args.max_steps

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

def check_low_extend(check_ex):
    if check_ex < 0:
        return 0
    else:
        return check_ex

def check_high_extend(check_ex, low_ex, orgsample_dim):
    if (low_ex + check_ex) > orgsample_dim:
        delta = (low_ex + check_ex) - orgsample_dim
        check_ex = check_ex - delta
        return check_ex
    else:
        return check_ex
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

        ## -- debug
        # # check how many samples:
        # total_samples = np.array(all_data[j][1]).shape[1]
        # print(np.array(all_data[j][1]).shape)
        # print('total samples: ', total_samples)
        # for k in range(total_samples):
        #     sample_left  = abs(int(all_data[j][1][1][k]))
        #     sample_top   = abs(int(all_data[j][1][2][k]))
        #     sample_width = abs(int(all_data[j][1][3][k]))
        #     sample_heigt = abs(int(all_data[j][1][4][k]))
        #     print('first: ', abs(int(all_data[j][1][0][k])))
        #     print('Sample left:   ', sample_left)
        #     print('Sample right:  ', sample_top)
        #     print('Sample width:  ', sample_width)
        #     print('Sample height: ', sample_heigt)
        # print('#--------------------------------------------------')

print('Completion..{%d/%d}' % (f['/digitStruct/bbox'].shape[0], f['/digitStruct/bbox'].shape[0]))
print('Completed!')

# Create copy of samples to curate labels
all_data_copy = deepcopy(all_data)
#-------------------------------------------------------------------------------

with open(curated_textfile, 'w') as ft:
    for sample_index in range(len(all_data)):
        try:
            sample_imgph = all_data[sample_index][0]
            sample_image = cv2.imread(file_path+sample_imgph)
            sample_image_copy = deepcopy(sample_image)
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
            low_x  = low_left - int(expand_percent * low_left)
            low_x  = check_low_extend(check_ex=low_x)

            low_y  = low_top  - int(expand_percent * low_top)
            low_y  = check_low_extend(check_ex=low_y)

            high_x = high_width  + int(expand_percent * high_width)
            high_x = check_high_extend(check_ex=high_x, low_ex=low_x,\
                                          orgsample_dim=sample_width_org)

            high_y = high_height + int(expand_percent * high_height)
            high_y = check_high_extend(check_ex=high_y, low_ex=low_y,\
                                          orgsample_dim=sample_heigt_org)

            #------- X-axis shift -------
            if low_x > 0:
                sample_image_copy = sample_image_copy[:, low_x:, :]
                # Update
                for i in range(total_samples):
                    # Update lower bound-- left
                    all_data_copy[sample_index][1][1][i] -= low_x

            if high_x < sample_width_org:
                sample_image_copy = sample_image_copy[:, :(low_x+high_x), :]

            #------- Y-axis shift-------
            if low_y > 0:
                sample_image_copy = sample_image_copy[low_y:, :, :]
                # Include fixed pixels from all upper bound-- top
                for i in range(total_samples):
                    all_data_copy[sample_index][1][2][i] -= low_y

            if high_y < sample_heigt_org:
                sample_image_copy = sample_image_copy[:(low_y+high_y), :, :]

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
                if index_into > total_samples - 1:
                    sample_label = 0
                    sample_left  = 0
                    sample_top   = 0
                    sample_width = 0
                    sample_heigt = 0

                else:
                    sample_label = int(all_data_copy[sample_index][1][0][index_into])
                    sample_left  = abs(int((all_data_copy[sample_index][1][1][index_into] * img_size[0])/sample_width_org_rz))
                    sample_top   = abs(int((all_data_copy[sample_index][1][2][index_into] * img_size[1])/sample_heigt_org_rz))
                    sample_width = abs(int((all_data_copy[sample_index][1][3][index_into] * img_size[0])/sample_width_org_rz))
                    sample_heigt = abs(int((all_data_copy[sample_index][1][4][index_into] * img_size[1])/sample_heigt_org_rz))

                # Append
                samples.append([all_data_copy[sample_index][0][:-4], sample_label, sample_left, sample_top, sample_width, sample_heigt])

            #-------------------------------------------------------------------
            ## Write and save
            # New sample Image path
            new_sample_image_path = os.path.join(curated_dataset, all_data_copy[sample_index][0])
            # Save
            cv2.imwrite(new_sample_image_path, smpl_img_rz)
            # Write
            ft.write(str(samples))
            ft.write('\n')

            #-------------------------------------------------------------------
            if sample_index%4000 == 0:
                print('Completion..{%d/%d}' % (sample_index, len(all_data)))

        except ValueError:
            print('Ignoring data: ', sample_imgph)

#------------------------------------------------------------------------------
# Close
ft.close()
print('Completion..{%d/%d}' % (len(all_data), len(all_data)))
print('Completed!')
#------------------------------------------------------------------------------
