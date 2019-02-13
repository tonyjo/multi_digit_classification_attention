from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.baseline_classify import Model
from src.data_loader_pred_classify import dataLoader

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
        _, predictions    = self.model.build_test_model()

        #------------------------Digit-Classification---------------------------
        # Reset graph
        tf.reset_default_graph()
        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            # Tensorboard summary path
            saver = tf.train.Saver()

            if self.pretrained_model is not None:
                print("Start testing with pretrained Model..")
                saver.restore(sess, self.pretrained_model)
            else:
                print("Start testing with Model with random weights...")

            lbl_batchs = []
            all_prdcts = []
            for i in range(n_iters):
                image_batch, label_batch = next(test_loader)
                # Convert one-hot to label
                feed_dict = {self.model.images: image_batch,
                             self.model.drop_prob: 1.0}
                # Predictions
                pred = sess.run(predictions, feed_dict)
                # Append
                all_prdcts.append(pred)
                lbl_batchs.append(label_batch)

                if i%self.print_every == 0:
                    print('Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')

        #----------------------Compute Sequence Accuracy------------------------
        total_acc = 0.0
        for t in range(len(lbl_batchs)):
            each_lbl_batch = lbl_batchs[t]
            each_prd_batch = all_prdcts[t]
            for T in range(len(each_lbl_batch)):
                each_labl_seq = each_lbl_batch[T]
                each_pred_seq = each_prd_batch[T]
                seq_accuracy  = 0
                for tT in range(len(each_labl_seq)):
                    each_labl = each_labl_seq[tT]
                    each_pred = each_pred_seq[tT]
                    if each_labl == each_pred:
                        seq_accuracy += 1
                # Get the accuracy of sequence
                seq_accuracy = seq_accuracy/len(each_labl_seq)
                # Update
                total_acc += seq_accuracy

        print('Final Sequence accuracy is: %f' % (total_acc/n_examples))
#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_cropped',
                      height=16, width=16, dataset_name='test.txt',
                      max_steps=7, mode='Test')
    # Load Model
    model = Model(image_height=16, image_width=16, l2=0.0002, mode='test')
    # Load Inference model
    testing = Test(model, data, batch_size=16, print_every=200,
                   pretrained_model='./model/clsfy1/model-640')
    # Begin Evaluation
    testing.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
