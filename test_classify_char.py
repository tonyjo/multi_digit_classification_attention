from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.baseline_classify import Model
from src.data_loader_classify import dataLoader

class Test(object):
    def __init__(self, model, data, **kwargs):
        self.model            = model
        self.data             = data
        self.batch_size       = kwargs.pop('batch_size', 64)
        self.print_every      = kwargs.pop('print_every', 100)
        self.pretrained_model = kwargs.pop('pretrained_model', None)


    def test(self):
        # Train dataset
        test_loader = self.data.gen_data_batch(self.batch_size)

        n_examples = self.data.max_length
        n_iters    = int(np.ceil(float(n_examples)/self.batch_size))

        # Summary
        print("Data size:  %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations: %d" %n_iters)

        # Build model
        _, predictions = self.model.build_test_model()
        ground_labels  = tf.placeholder(tf.float32, [None, 1])
        # Accuracy computation
        accuracy = tf.metrics.accuracy(labels=ground_labels, predictions=predictions)

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

            total_acc = 0.0

            for _ in range(n_iters):
                image_batch, label_batch = next(test_loader)
                # Convert one-hot to label
                label_batch = np.argmax(label_batch, axis=1)

                feed_dict = {self.model.images: image_batch,
                             ground_labels: label_batch,
                             self.model.drop_prob: 1.0}

                accu = sess.run(accuracy, feed_dict)

                total_acc += accu

        print('Final per-character accuracy is: %d %', % (total_acc/n_iters)*100)
#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_cropped',
                      height=16, width=16, dataset_name='test.txt',
                      max_steps=7, mode='Test')
    # Load Model
    model = Model(image_height=16, image_width=16, l2=0.0002, mode='test')

    # Load Inference model
    testing = Test(model, data, batch_size=1, print_every=500,
                   pretrained_model='None')
    # Begin Evaluation
    testing.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
