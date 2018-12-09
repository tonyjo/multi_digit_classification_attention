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
    def __init__(self, model, data, **kwargs):
        self.model            = model
        self.data             = data
        self.batch_size       = kwargs.pop('batch_size', 64)
        self.print_every      = kwargs.pop('print_every', 100)
        self.pretrained_model = kwargs.pop('pretrained_model', None)

    def bbox_iou_center_xy(self, bboxes1, bboxes2):
        """
        Args:
            bboxes1: shape (total_bboxes1, 4)
                with x1, y1, x2, y2 point order.
            bboxes2: shape (total_bboxes2, 4)
                with x1, y1, x2, y2 point order.

            p1 *-----
               |     |
               |_____* p2

        Returns:
            Tensor with shape (total_bboxes1, total_bboxes2)
            with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
            in [i, j].
        """

        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        xI2 = tf.minimum(x12, tf.transpose(x22))

        yI1 = tf.minimum(y11, tf.transpose(y21))
        yI2 = tf.maximum(y12, tf.transpose(y22))

        inter_area = tf.maximum((xI2 - xI1), 0) *\
                     tf.maximum((yI1 - yI2), 0)

        bboxes1_area = (x12 - x11) * (y11 - y12)
        bboxes2_area = (x22 - x21) * (y21 - y22)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

        return inter_area / (union + 0.0001)

    def test(self):
        # Train dataset
        test_loader = self.data.gen_data_batch(self.batch_size)

        n_examples = self.data.max_length
        n_iters    = int(np.ceil(float(n_examples)/self.batch_size))

        # Build model with IOU
        pred_bboxs_, alpha_list_, gnd_bboxs_ = self.model.build_test_model()
        interm_iou = []
        for t0 in range(3):
            iou = self.bbox_iou_center_xy(bboxes1=gnd_bboxs_[:, t0, :], bboxes2=pred_bboxs_[t0])
            interm_iou.append(iou)

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

            collect_iou = []
            for i in range(n_iters):
                image_batch, grd_bboxes_batch, _ = next(test_loader)

                feed_dict = {self.model.images: image_batch,
                             self.model.bboxes: grd_bboxes_batch,
                             self.model.drop_prob: 1.0}

                iou_out = sess.run(interm_iou, feed_dict)

                collect_iou.append(iou_out)

                if i%self.print_every == 0:
                    print('Epoch Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')

        # Close session
        sess.close()

        # Compute Final IOU score
        final_iou     = 0.0
        total_samples = 0.0
        collect_iou   = np.array(collect_iou)
        for t in range(n_iters):
            sample_iou = 0.0
            for T in range(3):
                sample_iou += collect_iou[t, T, 0, 0]
                total_samples += 1.0
            final_iou += sample_iou

        mean_iou = final_iou/total_samples

        print('The Final IOU score is: ', mean_iou)

#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_curated',
                      dataset_name='test.txt', max_steps=3, mode='Test')
    # Load Model
    model = Model(dim_feature=[49, 128], dim_hidden=128, n_time_step=3,
                  alpha_c=1.0, image_height=64, image_width=64, mode='test')
    # Load Trainer
    testing = Test(model, data, batch_size=1, print_every=1000, pretrained_model=None)
    # Begin Training
    testing.test()

if __name__ == "__main__":
    main()
