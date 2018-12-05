from __future__ import print_function
import os
import cv2

class dataLoader(object):
    def __init__(self, directory, dataset_name, max_steps, mode='Train'):
        self.max_steps    = max_steps
        self.directory    = directory
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

        self.all_data = all_data
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
            indices = range(len(self.images))
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
            grd_bboxes_batch = []
            grd_attnMk_batch = []
            # Generate training batch
            for _ in range(batch_size):
                sample_data = next(data_gen)
                sample_img_path = os.path.join(self.directory, self.dataset_name, sample_data[0][0])
                image = cv2.imread(sample_img_path)
                # Gather sample data for all time steps
                all_sample_data = []
                all_sample_attn = []
                for idx in range(self.max_steps):
                    # Get ground attention mask
                    sample_attn_mask_path = os.path.join(self.directory, self.dataset_name, sample_data[0][:-4] + '_' + str(idx) + '.npy')
                    sample_attn_mask = np.load(sample_attn_mask_path)

                    # Extract ground boxes-- sample_left, sample_top, sample_width, sample_height
                    sample_left   = abs(int(sample_data[idx][2]))
                    sample_top    = abs(int(sample_data[idx][3]))
                    sample_width  = abs(int(sample_data[idx][4]))
                    sample_height = abs(int(sample_data[idx][5]))

                    all_sample_attn.append(sample_attn_mask)
                    all_sample_data.append([sample_left, sample_top, sample_width, sample_height])

                # Append to generated batch
                image_batch.append(image)
                grd_bboxes_batch.append(all_sample_data)
                grd_attnMk_batch.append(all_sample_attn)

            yield np.array(image_batch), np.array(grd_bboxes_batch), np.array(grd_attnMk_batch)
