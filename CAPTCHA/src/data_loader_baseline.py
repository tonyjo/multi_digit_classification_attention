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

        # Get characters
        az = string.ascii_lowercase
        AZ = string.ascii_uppercase
        nm = string.digits
        # Append all characters
        all_selections = []
        for i in range(len(az)):
            all_selections.append(az[i])
        for i in range(len(AZ)):
            all_selections.append(AZ[i])
        for i in range(len(nm)):
            all_selections.append(nm[i])

        with open(file_path, 'r') as f:
            frames = f.readlines()

        for i in range(0, len(frames), self.max_steps):
            interm_data = []
            for u in range(self.max_steps):
                frame = frames[i+u]
                path, label, w1, h1, w2, h2 = frame.split(', ')
                h2 = h2[:-1] # Remove /n at the end
                if self.mode == 'Test':
                    path = int(path[0:-4]) + 55000
                    path = str(path) + '.png'
                if self.mode == 'Valid':
                    path = int(path[0:-4]) + 40000
                    path = str(path) + '.png'
                label = all_selections.index(label) # Convert to label category
                interm_data.append([path, label, int(w1), int(h1), int(w2), int(h2)])

            final_data = []
            for k in range(self.max_steps):
                if k == 0:
                    final_data.append(interm_data[0][0])
                    final_data.append(interm_data[k][1:])
                else:
                    final_data.append(interm_data[k][1:])
            all_data.append(final_data)

        self.all_data = all_data
        self.max_length = len(self.all_data)
        self.possible_pred = len(all_selections)

        print('All data Loaded!')

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
            # Generate training batch
            for _ in range(batch_size):
                sample_data     = next(data_gen)
                sample_img_path = os.path.join(self.directory, self.dataset_dir, sample_data[0])
                image           = cv2.imread(sample_img_path)
                image           = cv2.resize(image, (self.image_width, self.image_height))
                # if self.mode != 'Train':
                #     image_batch.append(image)
                # image_norm      = deepcopy(image)
                # Set image between -1 and 1
                image_norm = image /127.5 - 1.0
                # Gather sample data for all time steps
                all_sample_labl = []
                for idx in range((self.max_steps)):
                    # Extract and create ground labels
                    sample_label  = int(sample_data[idx][0])
                    one_hot_label = np.zeros(self.possible_pred)
                    one_hot_label[sample_label] = 1.0
                    # Collect
                    all_sample_labl.append(one_hot_label)
                # Append to generated batch
                image_norm_batch.append(image_norm)
                grd_lables_batch.append(all_sample_labl)

            yield np.array(image_norm_batch), np.array(grd_lables_batch)
