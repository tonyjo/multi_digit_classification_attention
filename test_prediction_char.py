from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.model import Model as Attn_Model
from src.baseline_classify import Model as Clfy_Model
from src.data_loader_pred_classify import dataLoader

class Test(object):
    def __init__(self, data, attn_model, clfy_model, img_size, **kwargs):
        self.data        = data
        self.width       = img_size[0]
        self.height      = img_size[1]
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
                # Ignore first and last prediction
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
            interm_images_crop_resize = []
            # Loop through predictions
            for bbx in bbox:
                sample_left   = abs(int(bbx[0]))
                sample_top    = abs(int(bbx[1]))
                sample_width  = abs(int(bbx[2]))
                sample_height = abs(int(bbx[3]))

                image_patch = image[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width, :]
                # Zooming
                image_patch_rz = cv2.resize(image_patch, (self.width, self.height), interpolation = cv2.INTER_AREA)
                # Normalize image between -1 and 1
                image_patch_rz = image_patch_rz /127.5 - 1.0
                # Collect
                interm_images_crop_resize.append(image_patch_rz)
            # Append to main
            images_crop_resize.append(interm_images_crop_resize)

        return images_crop_resize

    def test(self):
        # Train dataset
        test_loader = self.data.gen_data_batch(self.batch_size)
        n_examples  = self.data.max_length
        n_iters     = int(np.ceil(float(n_examples)/self.batch_size))
        # Total Char
        total_char = 0.0
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
                images_crop_rez = self.crop_and_resize(images=image_batch, bboxes=vald_prdct_bbxs)
                images_crop_rez = np.array(images_crop_rez)
                ## Digit classification
                for k in range(len(images_crop_rez)):
                    each_predt_images = np.array(images_crop_rez[k])
                    each_image_labels = label_batch[k]
                    # Check if prediction batch equals label_batch,
                    # if not select upto of the label batch
                    if len(each_predt_images) > len(each_image_labels):
                        each_predt_images = each_predt_images[0:len(each_image_labels), :, :, :]
                    elif len(each_predt_images) < len(each_image_labels):
                        each_predt_images_trim = each_predt_images[0:len(each_predt_images), :, :, :]
                    # Increment total characters using the orginal sequence not the trim
                    total_char += len(each_image_labels)
                    ## Digit Prediction
                    feed_dict = {self.clfy_model.images: each_predt_images_trim,
                                 self.clfy_model.labels: each_image_labels,
                                 self.clfy_model.drop_prob: 1.0}
                    # Run bounding
                    accu, pred = sess.run([accuracy, predictions], feed_dict)
                    total_acc += accu

                if i%self.print_every == 0:
                    print('Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')

        print('Final per-character accuracy is: %f' % (total_acc/total_char))
#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_curated',
                      dataset_name='test.txt', max_steps=7, mode='Test')
    # Load Attention Model
    attn_model = Attn_Model(dim_feature=[49, 128], dim_hidden=128, n_time_step=7,
                  alpha_c=1.0, image_height=64, image_width=64, mode='test')
    # Load
    clfy_model = Clfy_Model(image_height=16, image_width=16, l2=0.0002, mode='test')
    # Load Inference model
    testing = Test(data, attn_model, clfy_model, batch_size=16, img_size=(),\
                   print_every=2000, pretrained_clfy_model='model/lstm1/model-50',\
                   pretrained_attn_model='model/lstm1/model-50')
    # Begin Evaluation
    testing.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
