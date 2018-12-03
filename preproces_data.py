import os
import cv2
import h5py
import numpy as np
from copy import deepcopy

class SVHN_preprocess:
    def __init__(self, dataset_dir, max_steps, fixed_num, image_height,
                 image_width, ground_attention_downsample):
        """
        Args:
            dataset_dir : Path to the SVHN dataset
            max_steps   : Maximum number of digits to be recognized
            fixed_num   : Maximum image to include after the bounds
            image_height: Height to resize the image
            image_width : Width to resize the image
            ground_attention_downsample: tuple containing the resizing (width, height)
        """
        self.dataset_dir  = dataset_dir
        self.max_steps    = max_steps
        self.fixed_num    = fixed_num
        self.image_height = image_height
        self.image_width  = image_width
        self.ground_attention_downsample = ground_attention_downsample

    def get_name(self, index, hdf5_data):
        """
        Extracts the image file path from the mat file.
        """
        name = hdf5_data['/digitStruct/name']

        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

    def get_bbox(self, index, hdf5_data):
        """
        Extracts the ground truth bounding boxes of digits in each file,
        from the mat file
        """
        all_iterm_data = []
        item = hdf5_data['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = hdf5_data[item][key]
            values = [hdf5_data[attr.value[i].item()].value[0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            all_iterm_data.append(values)

        return all_iterm_data

    def extract_data(self, dataset_type):
        """
        Extracts the ground truth bounding boxes of digits in each file,
        from the mat file
        """
        mat_file = os.path.join(self.dataset_dir, dataset_type, 'digitStruct.mat')
        f = h5py.File(mat_file, 'r')
        print('Total file size: ', f['/digitStruct/name'].shape[0])
        print('Data Extraction in progress....')
        all_data = []
        for j in range(f['/digitStruct/bbox'].shape[0]):
            img_name = get_name(j, f)
            row_dict = get_bbox(j, f)

            all_data.append([img_name, row_dict])

            if j%4000 == 0:
                print('Completion..{%d/%d}' % (j, f['/digitStruct/bbox'].shape[0]))

        print('Completion..{%d/%d}' % (f['/digitStruct/bbox'].shape[0], f['/digitStruct/bbox'].shape[0]))
        print('Completed!')

        return all_data

    def gaussian2d(self, sup, scales):
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

    def generate_ground_attention_mask(self, sample, sample_top, sample_height, sample_left, sample_width):
        """
        Creates a ground truth attention mask based on ground truth bounding boxes,
        and scales to fit into the box.
        """
        scales = np.sqrt(2) * 8
        gaussian = gaussian2d((sample_width, sample_height), scales)
        gaussain_normalized = (gaussain - np.min(gaussain))/\
                              (np.max(gaussain) - np.min(gaussain))

        sample_attention = np.zeros_like(sample) * 0.0
        sample_attention = sample_attention[:, :, 0]

        sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = gaussain_normalized

        sample_attention_res = cv2.resize(sample_attention, self.ground_attention_downsample, interpolation=cv2.INTER_NEAREST)

        return sample_attention, sample_attention_res

    def get_bounds(self, all_data, sample_index):
        """
        Gets the upper and lower limit of the bounding box,
        of all digits in the image. This is done to prevent squeezing the digits,
        when resized to a smaller dimension.
        """
        lower_x = 1000
        upper_x = 0
        lower_y = 1000
        upper_y = 0

        # check how many samples:
        total_samples = np.array(all_data[sample_index][1]).shape[1]

        for i in range(total_samples):
            sample_left  = int(all_data[sample_index][1][1][i])
            sample_top   = int(all_data[sample_index][1][2][i])
            sample_width = int(all_data[sample_index][1][3][i])
            sample_heigt = int(all_data[sample_index][1][4][i])

            if sample_left < lower_x:
                lower_x = sample_left

            if sample_left + sample_width > upper_x:
                upper_x = sample_left + sample_width

            if sample_top < lower_y:
                lower_y = sample_top

            if sample_top + sample_heigt > lower_y:
                upper_y = sample_top + sample_heigt

        return lower_x, upper_x, lower_y, upper_y

    def crop_sample_using_bounds(self, all_data_copy, sample_index, sample_image):
        """
        Curates the dataset to fit the training/testing procedure
        """
        sample_height_org, sample width_org, _ = sample_image.shape
        sample_image_copy = deepcopy(sample_image)
        lower_x, upper_x, lower_y, upper_y = self.get_bounds(all_data_copy, sample_index)
        # check how many samples:
        total_samples = np.array(all_data[sample_index][1]).shape[1]

        if lower_x - self.fixed_num > 0:
            start = lower_x - self.fixed_num
            sample_image_copy = sample_image_copy[:, start:, :]
            # Subtract 40 from all lower bound-- left
            for i in range(total_samples):
                all_data_copy[sample_index][1][1][i] -= start
            upper_x_copy = upper_x - start
        else:
            upper_x_copy = upper_x

        if upper_x + fixed_num < sample_width_org:
            sample_image_copy = sample_image_copy[:, :upper_x_copy + fixed_num, :]

        if lower_y - fixed_num > 0:
            end = lower_y - fixed_num
            sample_image_copy = sample_image_copy[end:, :, :]
            # Subtract 40 from all lower bound-- top
            for i in range(total_samples):
                all_data_copy[sample_index][1][2][i] -= end
            upper_y_copy = upper_y - end
        else:
            upper_y_copy = upper_y

        if upper_y + fixed_num > sample_heigt_org:
            sample_image_copy = sample_image_copy[:upper_y_copy + fixed_num, :, :]

        return sample_image_copy, all_data_copy

    def rescale_img_and_grd_truth(self, sample_image, all_data_copy):
        """
        Curates the dataset to fit the training/testing procedure
        """
        sample_height, sample_width, _ = sample_image.shape
        # Resize sample_image
        sample_image_rz = cv2.resize(sample_image_copy, (self.image_width, image_height))
        # Check how many samples
        total_samples = np.array(all_data[sample_index][1]).shape[1]

        if total_samples >= self.max_steps:
            total_samples = self.max_steps
            # Collect samples
            samples = []
            samples_attention = []
            for index in range(total_samples):
                sample_label  = int( all_data_copy[sample_index][1][0][index])
                sample_left   = int((all_data_copy[sample_index][1][1][index] * self.image_width)/sample_width)
                sample_top    = int((all_data_copy[sample_index][1][2][index] * self.image_height)/sample_height)
                sample_width  = int((all_data_copy[sample_index][1][3][index] * self.image_width)/sample_width)
                sample_height = int((all_data_copy[sample_index][1][4][index] * self.image_height)/sample_height)
                sample_attention, sample_attention_res = self.generate_ground_attention_mask(sample_image_rz, sample_top, sample_height, sample_left, sample_width)
                # Append
                samples_attention.append([sample_attention])
                samples_attention_rz.append([sample_attention_res])
                samples.append([sample_label, sample_left, sample_top, sample_width, sample_height])
        else:
            # Collect samples
            prev_idx = 0
            samples  = []
            samples_attention = []
            samples_attention_rz = []
            for index in range(self.max_steps):
                # Duplicate samples
                if index + 1 <= total_samples:
                    prev_idx = index

                sample_label  = int( all_data_copy[sample_index][1][0][prev_idx])
                sample_left   = int((all_data_copy[sample_index][1][1][prev_idx] * self.image_width)/sample_width)
                sample_top    = int((all_data_copy[sample_index][1][2][prev_idx] * self.image_height)/sample_height)
                sample_width  = int((all_data_copy[sample_index][1][3][prev_idx] * self.image_width)/sample_width)
                sample_height = int((all_data_copy[sample_index][1][4][prev_idx] * self.image_height)/sample_height)
                # Append
                samples_attention.append([sample_attention])
                samples_attention_rz.append([sample_attention_res])
                samples.append([sample_label, sample_left, sample_top, sample_width, sample_height])

        return sample_image_rz, samples, samples_attention, samples_attention_rz

    def generate_data(self, dataset_type):
        """
        Curates the dataset to fit the training/testing procedure
        """
        curated_dataset  = os.path.join(self.dataset_dir, dataset_type + '_curated')
        curated_textfile = os.path.join(self.dataset_dir, dataset_type + '.txt')
        if os.path.exists(curated_dataset) == False:
            os.mkdir(curated_dataset)
        # Extract data from mat file
        all_dataset   = self.extract_data(dataset_type)
        # Create copy of data
        all_data_copy = deepcopy(all_dataset)

        with open(curated_textfile, 'w') as ft:
            for idx in range(len(all_dataset)):
                # Sample Image path
                sample_image_path = os.path.join(self.dataset_dir, dataset_type, all_dataset[idx][0])
                # Read the image
                sample_image = cv2.imread(sample_image_path)
                # Convert BGR to RGB
                sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
                # Crop image based on fixed length
                sample_image_crop, all_data_copy = self.crop_sample_using_bounds(all_data_copy, sample_index=idx, sample_image)
                # Resize and Rescale image and bounding boxes
                sample_image_rz, samples, samples_attention, samples_attention_rz = rescale_img_and_grd_truth(sample_image_crop, all_data_copy)
                # New sample Image path
                new_sample_image_path = os.path.join(curated_dataset, all_dataset[idx][0])
                # Save
                cv2.imwrite(new_sample_image_path, sample_image_rz)
                for j in range(max_steps):
                    # Sample Image path -- remove png and add npy
                    sample_attn_image_path = os.path.join(self.dataset_dir, dataset_type, all_dataset[idx][0][:-4] + '_' + str(j) + '.npy')
                    np.save(sample_attn_image_path, samples_attention_rz[j][0])
                # Write
                ft.write(samples)
                ft.write('\n')

            if j%4000 == 0:
                print('Completion..{%d/%d}' % (j, len(all_dataset)))
        # Close
        ft.close()
        print('Completion..{%d/%d}' % (len(all_dataset), len(all_dataset)))
        print('Completed!')


if __name__ == '__main__':
    # Arguments
    dataset_dir  = './datasets'
    max_steps    = 3
    image_height = 64
    image_width  = 64
    fixed_num    = 20
    ground_attention_downsample = (7, 7)

    svhn_preproces = SVHN_preprocess(dataset_dir, max_steps, fixed_num, image_height,
                                     image_width, ground_attention_downsample)
    # Generate Training Data
    svhn_preproces.generate_data(dataset_type='train')
    # Generate Testing Data
    svhn_preproces.generate_data(dataset_type='test')
