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
    def __init__(self, data, attn_model, clfy_model, **kwargs):
        self.data        = data
        self.attn_model  = attn_model
        self.clfy_model  = clfy_model
        self.batch_size  = kwargs.pop('batch_size', 64)
        self.print_every = kwargs.pop('print_every', 100)
        self.pretrained_clfy_model = kwargs.pop('pretrained_clfy_model', None)
        self.pretrained_attn_model = kwargs.pop('pretrained_attn_model', None)

    def valid_pred_bboxs(pred_bboxs_):
        pass

    def test(self):
        # Train dataset
        test_loader = self.data.gen_data_batch(self.batch_size)
        n_examples  = self.data.max_length
        n_iters     = int(np.ceil(float(n_examples)/self.batch_size))

        # Summary
        print("Data size:  %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations: %d" %n_iters)

        # Build model
        pred_bboxs_, _    = self.attn_model.build_test_model()
        _, predictions    = self.clfy_model.build_test_model()
        check_predictions = tf.equal(predictions, tf.argmax(self.model.labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(check_predictions, tf.float32))

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            # Tensorboard summary path
            saver = tf.train.Saver()

            if self.pretrained_clfy_model is not None:
                print("Start testing with Classification pretrained Model..")
                saver.restore(sess, self.pretrained_clfy_model)
            else:
                print("Start testing with Classification Model with random weights...")

            if self.pretrained_attn_model is not None:
                print("Start testing with pretrained Attention Model..")
                saver.restore(sess,self.pretrained_attn_model)
            else:
                print("Start testing with Attention Model with random weights...")

            total_acc = 0.0

            for i in range(n_iters):
                image_batch, label_batch = next(test_loader)
                # Convert one-hot to label
                feed_dict = {self.model.images: image_batch,
                             self.model.labels: label_batch,
                             self.model.drop_prob: 1.0}

                accu, pred = sess.run([accuracy, predictions], feed_dict)
                total_acc += accu

                if i%self.print_every == 0:
                    print('Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')

        print('Final per-character accuracy is: %f' % (total_acc/n_iters))
#-------------------------------------------------------------------------------
def main():
    pass
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
