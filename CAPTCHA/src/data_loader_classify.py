from __future__ import print_function
import os
import cv2
import string
import random
import numpy as np

class dataLoader(object):
    def __init__(self, directory, dataset_dir, dataset_name, max_steps,
                 image_width, image_height, image_patch_width, image_patch_height,
                 grd_attn=False, mode='Train'):
        self.mode         = mode
        self.grd_attn     = grd_attn
        self.max_steps    = max_steps
        self.image_width  = image_width
        self.image_height = image_height
        self.directory    = directory
        self.dataset_dir  = dataset_dir
        self.dataset_name = dataset_name
        self.image_patch_width = image_patch_height
        self.image_patch_height = image_patch_height
        self.load_data()

    def load_data(self):
        all_data = []
        # Full images file path
        file_path = os.path.join(self.directory, self.dataset_name)
        #-----------------------------------------------------------------------
        # Get characters
        az = string.ascii_lowercase
        AZ = string.ascii_uppercase
        nm = string.digits
        #-----------------------------------------------------------------------
        # Append all characters
        all_selections = []
        for i in range(len(az)):
            all_selections.append(az[i])
        for i in range(len(AZ)):
            all_selections.append(AZ[i])
        for i in range(len(nm)):
            all_selections.append(nm[i])
        #-----------------------------------------------------------------------
        with open(file_path, 'r') as f:
            frames = f.readlines()

        for i in range(0, len(frames), self.max_steps):
            for u in range(self.max_steps):
                frame = frames[i+u]
                path, label, w1, h1, w2, h2 = frame.split(', ')
                h2 = h2[:-1] # Remove /n at the end
                label_num = all_selections.index(label) # Convert to label category
                all_data.append([path, int(label_num), int(w1), int(h1), int(w2), int(h2)])

        self.all_data = all_data
        self.max_length = len(self.all_data)
        self.possible_pred = len(all_selections)

        print('All data Loaded!')

    def randomFlip(self, image):
        flip_p = np.random.rand()
        if flip_p > 0.5:
            flip_image = image[:, ::-1]
        else:
            flip_image = image

        return flip_image

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
        # Loop
        while True:
            image_batch = []
            label_batch = []
            # Generate training batch
            for _ in range(batch_size):
                valid = None
                while valid is None:
                    sample_data = next(data_gen)
                    sample_img_path = os.path.join(self.directory, self.dataset_dir, sample_data[0])
                    try:
                        image = cv2.imread(sample_img_path)
                        org_img_hgth, org_img_wdth, _ = image.shape
                        image = cv2.resize(image, (self.image_width, self.image_height))
                        # Gather sample data
                        # path, label, int(w1), int(h1), int(w2), int(h2)
                        sample_label  = sample_data[1]
                        #print(sample_label)
                        one_hot_label = np.zeros(self.possible_pred)
                        one_hot_label[sample_label] = 1.0
                        # Get Bboxes
                        sample_left   = sample_data[2] * 1.0
                        sample_top    = sample_data[3] * 1.0
                        sample_width  = sample_data[4] * 1.0
                        sample_height = sample_data[5] * 1.0
                        # Rescale axis to resized image
                        sample_left   = np.ceil((sample_left   * self.image_width)/org_img_wdth)
                        sample_top    = np.ceil((sample_top    * self.image_height)/org_img_hgth)
                        sample_width  = np.floor((sample_width  * self.image_width)/org_img_wdth)
                        sample_height = np.floor((sample_height * self.image_height)/org_img_hgth)
                        # Extract image_patch
                        image_patch = image[int(sample_top):int(sample_top+sample_height),\
                                            int(sample_left):int(sample_left+sample_width), :]
                        # Resize
                        image_patch_rz = cv2.resize(image_patch, (self.image_patch_width, self.image_patch_height),\
                                                    interpolation = cv2.INTER_LINEAR)
                        # Data Augmentation
                        # image_patch_rz = self.randomFlip(image_patch_rz)
                        # Set image patch between -1 and 1
                        image_patch_rz = image_patch_rz/127.5 - 1.0
                        # Append to generated batch
                        image_batch.append(image_patch_rz)
                        label_batch.append(one_hot_label)
                        # Set valid flag to not None
                        valid = 'True'
                    except cv2.error as e:
                        print('File error at: ', sample_img_path, ' resampling..')

            yield np.array(image_batch), np.array(label_batch)
