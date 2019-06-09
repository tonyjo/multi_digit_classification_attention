from __future__ import print_function
import os
import cv2
import string
import random
import numpy as np
from copy import deepcopy

class dataLoader(object):
    def __init__(self, directory, dataset_dir, dataset_name, max_steps,
                 image_width, image_height, grd_attn=False, mode='Train'):
        self.mode         = mode
        self.grd_attn     = grd_attn
        self.max_steps    = max_steps
        self.image_width  = image_width
        self.image_height = image_height
        self.directory    = directory
        self.dataset_dir  = dataset_dir
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self):
        all_data = []
        # Full images file path
        file_path = os.path.join(self.directory, self.dataset_name)

        with open(file_path, 'r') as f:
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
            for u in range(self.max_steps):
                each_data = []
                for k in range(6):
                    if k == 0:
                        each_data.append(str(iterm_data[count]) + '.png')
                    else:
                        each_data.append(iterm_data[count])
                    count += 1
                final_data.append(each_data)

            all_data.append(final_data)

        self.all_data   = all_data
        self.max_length = len(self.all_data)
        self.possible_pred = len(all_selections)

        print('All data Loaded!')

    def gaussian2d(self, sup, scales):
        """
        Creates a 2D Gaussian based on the size and scale.
        """
        var   = scales * scales
        shape = (sup[0], sup[1])
        n,m   = [(i-1)/2 for i in shape]
        y,x   = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var))

        return g

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum(axis=0)

    def generate_ground_gaussian_attention_mask(self, sample_size, sample_top,\
                                      sample_height, sample_left, sample_width):
        """
        Creates a ground truth attention mask based on ground truth bounding boxes,
        and scales to fit into the box.
        """
        sample_image_height, sample_image_width = sample_size
        scales = np.sqrt(2) * 10 # Play with the standard deviation

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
        gaussain = self.gaussian2d((sample_width, sample_height), scales)
        gaussain_normalized = (gaussain - np.min(gaussain))/\
                              (np.max(gaussain) - np.min(gaussain))
        h_gauss_norm, w_gauss_norm = gaussain_normalized.shape
        gaussain_normalized = gaussain_normalized.flatten()
        gaussain_normalized = self.softmax(gaussain_normalized)
        gaussain_normalized = np.reshape(gaussain_normalized, (h_gauss_norm, w_gauss_norm))

        sample_attention = np.zeros((sample_image_height, sample_image_width)) * 0.0

        sample_attention[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width] = gaussain_normalized

        return sample_attention

    def generate_start_attention_mask(self, attn_size):
        """
        Create an attention mask for the start state.
        """
        x = np.zeros(attn_size) * 0.0
        x[0, 0] = 1.0

        return x

    def generate_stop_attention_mask(self, attn_size):
        """
        Create an attention mask for the stop state.
        """
        x = np.zeros(attn_size) * 0.0
        x[-1, -1] = 1.0

        return x

    def gen_random_data(self):
        while True:
            indices = list(range(len(self.all_data)))
            random.shuffle(indices)
            for i in indices:
                data  = self.all_data[i]

                yield data

    def gen_val_data(self):
        while True:
            indices = range(len(self.all_data))
            for i in indices:
                data  = self.all_data[i]

                yield data

    def gen_data_batch(self, batch_size):
        ground_attention_size = (14, 14) # (height, width)
        # Pre-generate start and stop state attention mask
        if self.grd_attn == True:
            start_attn_mask = self.generate_start_attention_mask(attn_size=ground_attention_size)
            stop_attn_mask  = self.generate_stop_attention_mask(attn_size=ground_attention_size)
        # Generate data based on training/validation
        if self.mode == 'Train':
            # Randomize data
            data_gen = self.gen_random_data()
        else:
            # Validation Data generation
            data_gen = self.gen_val_data()

        while True:
            image_batch      = []
            image_batch_norm = []
            grd_lables_batch = []
            grd_bboxes_batch = []
            grd_attnMk_batch = []
            # Generate training batch
            for _ in range(batch_size):
                sample_data     = next(data_gen)
                sample_img_path = os.path.join(self.directory, self.dataset_dir, sample_data[0])
                image           = cv2.imread(sample_img_path)
                org_img_hgth, org_img_wdth, _ = image.shape
                # if self.mode != 'Train':
                #     image_batch.append(image)
                # image_norm      = deepcopy(image)
                # Set image between -1 and 1
                image_norm = image /127.5 - 1.0
                # Gather sample data for all time steps
                all_sample_data = []
                all_sample_attn = []
                all_sample_labl = []
                for idx in range((self.max_steps)):
                    # Extract ground boxes-- sample_left, sample_top, sample_width, sample_height
                    sample_label  = int(sample_data[idx][0])
                    sample_left   = sample_data[idx][1] * 1.0
                    sample_top    = sample_data[idx][2] * 1.0
                    sample_width  = sample_data[idx][3] * 1.0
                    sample_height = sample_data[idx][4] * 1.0
                    # Start State
                    if sample_label == 0:
                        one_hot_label = np.zeros(12)
                        one_hot_label[sample_label] = 1.0
                        sample_left   = 0.0
                        sample_top    = 0.0
                        sample_width  = 1.0
                        sample_height = 1.0
                        if self.grd_attn == True:
                            attn_mask = start_attn_mask
                    # End State
                    elif sample_label == 11:
                        one_hot_label = np.zeros(12)
                        one_hot_label[sample_label] = 1.0
                        sample_left   = self.image_width - 3
                        sample_top    = self.image_height - 3
                        sample_width  = 1.0
                        sample_height = 1.0
                        if self.grd_attn == True:
                            attn_mask = stop_attn_mask
                    else:
                        # Extract and create ground labels
                        one_hot_label = np.zeros(12)
                        one_hot_label[sample_label] = 1.0
                        if self.grd_attn == True:
                            # Then rescale axis to the final feature map size
                            sple_left   = int(np.ceil((sample_left    * ground_attention_size[1])/self.image_width))
                            sple_top    = int(np.ceil((sample_top     * ground_attention_size[0])/self.image_height))
                            sple_width  = int(np.floor((sample_width  * ground_attention_size[1])/self.image_width))
                            sple_heigth = int(np.floor((sample_height * ground_attention_size[0])/self.image_height))
                            # Generate mask
                            attn_mask   = self.generate_ground_gaussian_attention_mask(ground_attention_size,\
                                                         sple_top, sple_heigth, sple_left, sple_width)
                    # Collect
                    if self.grd_attn == True:
                        attn_mask = attn_mask.flatten()
                        all_sample_attn.append(attn_mask)
                    all_sample_data.append([sample_left, sample_top, sample_width, sample_height])
                    all_sample_labl.append(one_hot_label)
                # Append to generated batch
                # image_batch.append(image)
                if self.grd_attn == True:
                    grd_attnMk_batch.append(all_sample_attn)
                image_batch_norm.append(image_norm)
                grd_bboxes_batch.append(all_sample_data)
                grd_lables_batch.append(all_sample_labl)

            if self.grd_attn == True:
                # yield np.array(image_batch), np.array(image_batch_norm), np.array(grd_bboxes_batch), np.array(grd_attnMk_batch)
                yield np.array(image_batch_norm), np.array(grd_lables_batch), np.array(grd_bboxes_batch), np.array(grd_attnMk_batch)
            else:
                # yield np.array(image_batch), np.array(grd_lables_batch), np.array(image_batch_norm), np.array(grd_bboxes_batch)
                yield np.array(image_batch_norm), np.array(grd_lables_batch), np.array(grd_bboxes_batch)
