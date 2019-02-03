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
        self.max_steps   = kwargs.pop('max_steps', 7)
        self.batch_size  = kwargs.pop('batch_size', 16)
        self.print_every = kwargs.pop('print_every', 100)
        self.pretrained_clfy_model = kwargs.pop('pretrained_clfy_model', None)
        self.pretrained_attn_model = kwargs.pop('pretrained_attn_model', None)

    def valid_pred_bboxs(self, pred_bboxs):
        """
        The first step and last step are start and stop state
        """
        def box_threshold(left, top, width, height):
            valid_box = False
            # If the threshold box is less than 10
            if width * height < 10:
                valid_box = False
            elif int(left) == 0 and int(top) == 0 and\
                 int(width) == 0 and int(height) == 0:
                valid_box = False
            else:
                valid_box = True

            return valid_box

        valid_pred_bboxs = []
        for t in range(len(pred_bboxs)):
            interm_pred_bboxs = []
            for T in range(self.max_steps):
                if T != 0 and T+1 != self.max_steps:
                    # Predicted bounding box
                    smple_left   = pred_bboxes[t][T][0][0]
                    smple_top    = pred_bboxes[t][T][0][1]
                    smple_width  = pred_bboxes[t][T][0][2]
                    smple_height = pred_bboxes[t][T][0][3]
                    # Check if the box is valid
                    vld_bbx_pred = box_threshold(left=smpl_left, top=smple_top,\
                                         width=smple_width, height=smple_height)
                    # Collect if valid box
                    if vld_bbx_pred:
                        interm_pred_bboxs.append([smple_left, smple_top, smple_width, smple_height])
            # Collect
            valid_pred_bboxs.append(interm_pred_bboxs)

        return valid_pred_bboxs

    def crop_and_resize(self, images, bboxes):
        images_crop_resize = []
        for t in range(len(bboxes)):
            image = images[t]
            bbox  = bboxes[t]

            for bbx in bbox:
                sample_left   = abs(int(bbx[0]))
                sample_top    = abs(int(bbx[1]))
                sample_width  = abs(int(bbx[2]))
                sample_height = abs(int(bbx[3]))

                image_patch = image[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width, :]
                # Zooming
                image_patch_rz = cv2.resize(image_patch, (self.width, self.height), interpolation = cv2.INTER_AREA)
                # Set image between -1 and 1
                image_patch_rz = image_patch_rz /127.5 - 1.0

                images_crop_resize.append(image_patch_rz)

        return images_crop_resize

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
        # Classification score
        check_predictions = tf.equal(predictions, tf.argmax(self.clfy_model.labels, axis=1))
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
                image_batch, image_norm_batch, label_batch = next(test_loader)

                ## Box Prediction
                feed_dict = {self.attn_model.images: image_norm_batch,
                             self.attn_model.drop_prob: 1.0}
                # Run bounding box prediction
                predicted_bboxs = sess.run(pred_bboxs_, feed_dict)
                vald_prdct_bbxs = self.valid_pred_bboxs(pred_bboxs=predicted_bboxs)
                images_crop_rez = self.crop_and_resize(images=, bboxes=vald_prdct_bbxs)
                images_crop_rez = np.array(images_crop_rez)

                ## Digit classification
                # Check if prediction batch == label_batch,
                # if not select upto of the label batch
                if len(images_crop_rez) != len(label_batch):
                    images_crop_rez = images_crop_rez[0:len(label_batch), :, :, :]

                feed_dict = {self.clfy_model.images: images_crop_rez,
                             self.clfy_model.labels: label_batch,
                             self.clfy_model.drop_prob: 1.0}

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
