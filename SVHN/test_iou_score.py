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
        self.max_steps        = kwargs.pop('max_steps', 6)
        self.batch_size       = kwargs.pop('batch_size', 64)
        self.print_every      = kwargs.pop('print_every', 100)
        self.pretrained_model = kwargs.pop('pretrained_model', None)

    def bb_intersection_over_union(self, boxA, boxB):
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
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def test(self):
        # Train dataset
        test_loader = self.data.gen_data_batch(self.batch_size)

        n_examples = self.data.max_length
        n_iters    = int(np.ceil(float(n_examples)/self.batch_size))

        # Build model with IOU
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

            collect_iou = []
            collect_grnd_bboxes = []
            for i in range(n_iters):
                image_batch, grd_bboxes_batch, _ = next(test_loader)

                feed_dict = {self.model.images: image_batch,
                             self.model.bboxes: grd_bboxes_batch,
                             self.model.drop_prob: 1.0}

                pred_bboxs_out = sess.run(pred_bboxs_, feed_dict)

                collect_iou.append(pred_bboxs_out)
                collect_grnd_bboxes.append(grd_bboxes_batch)

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
        collect_grnd_bboxes = np.array(collect_grnd_bboxes)

        # print(collect_iou.shape)
        # print(collect_grnd_bboxes.shape)

        for t in range(n_iters):
            sample_iou = 0.0
            for T in range(3):
                iterm_pred_bboxes = collect_iou[t, T, 0, :]
                iterm_grnd_bboxes = collect_grnd_bboxes[t, 0, T, :]

                leftA, topA, widthA, heightA = iterm_grnd_bboxes[0], iterm_grnd_bboxes[1],\
                                               iterm_grnd_bboxes[2], iterm_grnd_bboxes[3]
                leftB, topB, widthB, heightB = iterm_pred_bboxes[0], iterm_pred_bboxes[1],\
                                               iterm_pred_bboxes[2], iterm_pred_bboxes[3],

                rightA = leftA + widthA
                downA  = topA  + heightA
                rightB = leftB + widthB
                downB  = topB  + heightB

                boxA = (leftA, topA, rightA, downA)
                boxB = (leftB, topB, rightB, downB)

                iterm_iout_sample = self.bb_intersection_over_union(boxA, boxB)
                sample_iou += iterm_iout_sample
                total_samples += 1.0

            final_iou += sample_iou

        mean_iou = final_iou/total_samples

        print('The Final Mean IOU score is: ', mean_iou)

#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_curated',
                      dataset_name='test.txt', max_steps=6, mode='Test')
    # Load Model
    model = Model(dim_feature=[196, 128], dim_hidden=128, n_time_step=6,
                  alpha_c=1.0, image_height=64, image_width=64, mode='test')
    # Load Inference model
    testing = Test(model, data, max_steps=6, batch_size=1, print_every=2000,
                   pretrained_model='model/lstm2/model-650')
    # Begin Evaluation
    testing.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
