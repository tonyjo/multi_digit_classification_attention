from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.model import Model
from src.data_loader import dataLoader

class Test(object):
    def __init__(self, model, data, val_data, **kwargs):
        self.model         = model
        self.data          = data
        self.val_data      = val_data
        self.batch_size    = kwargs.pop('batch_size', 64)
        self.print_every   = kwargs.pop('print_every', 100)
        self.log_path      = kwargs.pop('log_path', './log/')
        self.model_path    = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)

    def bbox_iou_center_xy(bboxes1, bboxes2):
        """
        same as `bbox_iou_corner_xy', except that we have
        center_x, center_y, w, h instead of x1, y1, x2, y2
        """

        x11, y11, w11, h11 = tf.split(bboxes1, 4, axis=1)
        x21, y21, w21, h21 = tf.split(bboxes2, 4, axis=1)

        xi1 = tf.maximum(x11, tf.transpose(x21))
        xi2 = tf.minimum(x11, tf.transpose(x21))

        yi1 = tf.maximum(y11, tf.transpose(y21))
        yi2 = tf.minimum(y11, tf.transpose(y21))

        wi = w11/2.0 + tf.transpose(w21/2.0)
        hi = h11/2.0 + tf.transpose(h21/2.0)

        inter_area = tf.maximum(wi - (xi1 - xi2 + 1), 0) *\
                     tf.maximum(hi - (yi1 - yi2 + 1), 0)

        bboxes1_area = w11 * h11
        bboxes2_area = w21 * h21

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

        return inter_area/(union + 0.0001)

    def test(self):
        # Train dataset
        n_examples = self.data.max_length
        n_iters    = int(np.ceil(float(n_examples)/self.batch_size))

        # Build model with loss
        pred_bboxs_, alpha_list_ = self.model.build_test_model()

        # Summary
        print("Data size:  %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations: %d" %n_iters)

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Tensorboard summary path
            saver = tf.train.Saver()

            if self.pretrained_model is not None:
                print("Start testing with pretrained Model..")
                saver.restore(sess, self.pretrained_model)
            else:
                print("Start testing with Model with random weights...")

            for i in range(n_iters):
                image_batch, grd_bboxes_batch, _= next(self.data.gen_data_batch(self.batch_size))
                feed_dict = {self.model.images: image_batch,
                             self.model.drop_prob: 1.0}

                _, prediction_bboxes = sess.run(pred_bboxs_, feed_dict)

                if i%self.print_every == 0:
                    print('Epoch Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')
        # Close session
        sess.close()
#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_curated', dataset_name='test.txt', max_steps=3, mode='Test')
    # Load Model
    model = Model(dim_feature=[49, 256], dim_hidden=128, n_time_step=3,
                  alpha_c=0.0, image_height=64, image_width=64, mode='test')
    # Load Trainer
    testing = Test(model, data, batch_size=64, print_every=100, pretrained_model=None)
    # Begin Training
    testing.test()

if __name__ == "__main__":
    main()
