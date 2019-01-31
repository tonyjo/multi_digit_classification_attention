from __future__ import print_function
import os
import cv2
import random
import numpy as np

class dataLoader(object):
    def __init__(self, directory, dataset_dir, height, width, dataset_name, max_steps, mode='Train'):
        self.mode         = mode
        self.height       = height
        self.width        = width
        self.max_steps    = max_steps
        self.directory    = directory
        self.dataset_dir  = dataset_dir
        self.dataset_name = dataset_name
        self.load_data()

    def convert_seq2char(self, data):
        all_data = []
        for sample_idx in range(len(data)):
            sample = data[sample_idx]
            for idx in range(len(sample)):
                sample_path  = sample[idx][0]
                sample_label = sample[idx][1]
                sample_left  = sample[idx][2]
                sample_top   = sample[idx][3]
                sample_width = sample[idx][4]
                sample_heigt = sample[idx][5]

                if sample_left == 0 and sample_top == 0 and sample_width == 0 and sample_heigt == 0:
                    continue
                elif sample_left == 1 and sample_top == 1 and sample_width == 1 and sample_heigt == 1:
                    continue
                else:
                    all_data.append([sample_path, sample_label, sample_left, sample_top, sample_width, sample_heigt])

        return all_data

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

        all_data_final  = self.convert_seq2char(data=all_data)
        self.all_data   = all_data_final
        self.max_length = len(self.all_data)

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
            image_batch = []
            label_batch = []
            # Generate training batch
            for _ in range(batch_size):
                sample_data = next(data_gen)
                sample_img_path = os.path.join(self.directory, self.dataset_dir, sample_data[0])
                image = cv2.imread(sample_img_path)
                # Gather sample data for all time steps
                # Extract ground boxes-- sample_left, sample_top, sample_width, sample_height
                sample_label  = abs(int(sample_data[1])) * 1.0
                #print(sample_label)
                one_hot_label = np.zeros(10) * 0.0
                if sample_label == 10:
                    one_hot_label[0] = 1.0
                else:
                    one_hot_label[int(sample_label)] = 1.0

                sample_left   = abs(int(sample_data[2]))
                sample_top    = abs(int(sample_data[3]))
                sample_width  = abs(int(sample_data[4]))
                sample_height = abs(int(sample_data[5]))

                image_patch = image[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width, :]
                # Zooming
                image_patch_rz = cv2.resize(image_patch, (self.width, self.height), interpolation = cv2.INTER_AREA)
                # Set image between -1 and 1
                image_patch_rz = image_patch_rz /127.5 - 1.0
                # Append to generated batch
                image_batch.append(image_patch_rz)
                label_batch.append(one_hot_label)

            yield np.array(image_batch), np.array(label_batch)
