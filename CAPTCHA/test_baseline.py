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
    def __init__(self, model, data, **kwargs):
        self.model         = model
        self.data          = data
        self.max_steps     = kwargs.pop('max_steps', 6)
        self.batch_size    = kwargs.pop('batch_size', 64)
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

    def test(self):
        # Test Data Loader
        valid_loader = self.data.gen_data_batch(self.batch_size)

        # Test dataset
        n_examples = self.data.max_length
        valid_n_iters = int(np.ceil(float(n_examples)/self.batch_size))
        print("Total Interations: %d" %valid_n_iters)

        # Build model with loss
        Z_L, Z_S1, Z_S2, Z_S3, Z_S4, Z_S5, Z_S6 = self.model.build_model()
        print('Build Model Success!')

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Load pretrained model
            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)
            # Var
            final_seq_acc_prd = 0.0
            # Run Loop
            for t in range(valid_n_iters):
                image_val_batch, grd_val_lables_batch, grd_val_mxstep_batch = next(valid_loader)
                feed_dict = {self.model.images: image_val_batch,
                             self.model.is_train: False,
                             self.model.drop_prob: 1.0}
                Z_L_pred, Z_S1_pred, Z_S2_pred, Z_S3_pred, Z_S4_pred, Z_S5_pred, Z_S6_pred = sess.run([Z_L, Z_S1, Z_S2, Z_S3, Z_S4, Z_S5, Z_S6], feed_dict)
                # Prediction
                Z_S1_pred = np.argmax(Z_S1_pred[0])
                Z_S2_pred = np.argmax(Z_S2_pred[0])
                Z_S3_pred = np.argmax(Z_S3_pred[0])
                Z_S4_pred = np.argmax(Z_S4_pred[0])
                Z_S5_pred = np.argmax(Z_S5_pred[0])
                Z_S6_pred = np.argmax(Z_S6_pred[0])
                # Label
                Z_S1_grnd = np.argmax(grd_val_lables_batch[0, 0, :])
                Z_S2_grnd = np.argmax(grd_val_lables_batch[0, 1, :])
                Z_S3_grnd = np.argmax(grd_val_lables_batch[0, 2, :])
                Z_S4_grnd = np.argmax(grd_val_lables_batch[0, 3, :])
                Z_S5_grnd = np.argmax(grd_val_lables_batch[0, 4, :])
                Z_S6_grnd = np.argmax(grd_val_lables_batch[0, 5, :])
                # Match
                sample_acc_prd = 0.0
                if Z_S1_pred == Z_S1_grnd:
                    sample_acc_prd += 1.0
                if Z_S2_pred == Z_S2_grnd:
                    sample_acc_prd += 1.0
                if Z_S3_pred == Z_S3_grnd:
                    sample_acc_prd += 1.0
                if Z_S4_pred == Z_S4_grnd:
                    sample_acc_prd += 1.0
                if Z_S5_pred == Z_S5_grnd:
                    sample_acc_prd += 1.0
                if Z_S6_pred == Z_S6_grnd:
                    sample_acc_prd += 1.0
                # Collect
                final_seq_acc_prd += sample_acc_prd/self.max_steps
                # Print every
                if t%4000 == 0:
                    print('Inference Completion..{%d/%d}' % (t, valid_n_iters))
            #-----------------------------------------------------
            print('Inference Completion..{%d/%d}' % (valid_n_iters, valid_n_iters))
            print('Completed!')
            # Prediction Accuracy
            print('Sequence Classification Accuracy: ',\
                      np.round((final_seq_acc_prd/valid_n_iters), 4) * 100, '%')
        # Close session
        sess.close()
#-------------------------------------------------------------------------------
def main():
    # Load train/val dataset
    data = dataLoader(directory='./dataset/captcha', dataset_dir='test',\
                      dataset_name='test.txt', max_steps=6, image_width=200,\
                      image_height=64, grd_attn=False, mode='Test')
    # Load Model
    model = Model_Baseline(image_height=64, image_width=200, mode='test')
    # Load Trainer
    Tester = Test(model, data=data, batch_size=1, max_steps=6, pretrained_model='model/baseline/')
    # Begin Training
    Tester.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
