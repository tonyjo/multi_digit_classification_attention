from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.baseline_classify import Model
from src.data_loader_seq_classify import dataLoader

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
        n_examples  = self.data.max_length
        n_iters     = int(np.ceil(float(n_examples)/self.batch_size))

        # Summary
        print("Data size:  %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations: %d" %n_iters)

        # Build model
        _, predictions  = self.model.build_test_model()

        #------------------------Digit-Classification---------------------------
        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Tensorboard summary path
            saver = tf.train.Saver()
            # Load pretrained model
            if self.pretrained_model is not None:
                print("Start testing with pretrained Model..")
                saver.restore(sess, self.pretrained_model)
            else:
                print("Start testing with Model with random weights...")
            # Run predictions
            lbl_batchs = []
            all_prdcts = []
            for i in range(n_iters):
                _, _, img_seq_batch, label_batch = next(test_loader)
                each_lbl = []
                each_prd = []
                for k in range(self.batch_size):
                    each_img_seq = img_seq_batch[k]
                    each_img_lbl = label_batch[k]
                    # Convert one-hot to label
                    feed_dict = {self.model.images: np.array(each_img_seq),
                                 self.model.drop_prob: 1.0}
                    # Predictions
                    pred = sess.run(predictions, feed_dict)
                    # Collect
                    each_prd.append(pred)
                    each_lbl.append(each_img_lbl)
                # Append
                all_prdcts.append(each_prd)
                lbl_batchs.append(each_lbl)
                # Print every
                if i%self.print_every == 0:
                    print('Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')

        #----------------------Compute Sequence Accuracy------------------------
        total_acc = 0.0
        total_seq = 0.0
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
                # Update sequence
                total_seq += 1
                # Update
                total_acc += seq_accuracy
        print('Final Sequence accuracy is: %f' % (total_acc/total_seq))
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
                   pretrained_model='./model/clsfy2/model-392')
    # Begin Evaluation
    testing.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
