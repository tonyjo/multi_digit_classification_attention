from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.data_loader_baseline import dataLoader
from src.model_baseline import Model_Baseline

class Test(object):
    def __init__(self, model, data, val_data, **kwargs):
        self.model         = model
        self.data          = data
        self.max_steps     = kwargs.pop('max_steps', 6)
        self.batch_size    = kwargs.pop('batch_size', 64)
        self.print_every   = kwargs.pop('print_every', 1000)
        self.log_path      = kwargs.pop('log_path', './log/')
        self.model_path    = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)

    def bbox_threshold(self, left, top, width, height):
        valid_box = False
        # If the threshold box is less than 30
        if width * height < 30:
            valid_box = False
        elif left == 0 and top == 0 and\
             width == 0 and height == 0:
            valid_box = False
        else:
            valid_box = True

        return valid_box

    def train(self):
        # Test Data Loader
        valid_loader = self.data.gen_data_batch(self.val_bth_size)

        # Test dataset
        n_examples = self.data.max_length
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))

        # Build model with loss
        Z_L, Z_S1, Z_S2, Z_S3, Z_S4, Z_S5, Z_S6 = self.model.build_model()


      # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Tensorboard summary path
            summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(max_to_keep=4)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)




