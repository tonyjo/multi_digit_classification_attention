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

    def valid_pred_bboxs(self, pred_bboxes):
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
        for t in range(self.batch_size):
            interm_pred_bboxs = []
            for T in range(max_steps):
                # Ignore first and last prediction
                if T != 0 and T+1 != max_steps:
                    # Predicted bounding box
                    smple_left   = abs(pred_bboxes[T][t][0])
                    smple_top    = abs(pred_bboxes[T][t][1])
                    smple_width  = abs(pred_bboxes[T][t][2])
                    smple_height = abs(pred_bboxes[T][t][3])
                    # Check if the box is valid
                    vld_bbx_pred = box_threshold(left=smple_left, top=smple_top,\
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
            image = images[t] # Sample image
            bbox  = bboxes[t] # Sample predictions
            interm_images_crop_resize = []
            # Loop through predictions
            for bbx in bbox:
                sample_left   = abs(int(bbx[0]))
                sample_top    = abs(int(bbx[1]))
                sample_width  = abs(int(bbx[2]))
                sample_height = abs(int(bbx[3]))
                # Crop from image
                image_patch = image[sample_top:sample_top+sample_height, sample_left:sample_left+sample_width, :]
                # Zooming
                image_patch_rz = cv2.resize(image_patch, (width, height), interpolation = cv2.INTER_AREA)
                # Normalize image between -1 and 1
                image_patch_rz = image_patch_rz /127.5 - 1.0
                # Collect
                interm_images_crop_resize.append(image_patch_rz)
            # Append to main
            images_crop_resize.append(interm_images_crop_resize)

        return images_crop_resize

    def test(self):
        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        # Test dataset
        test_loader = self.data.gen_data_batch(self.batch_size)
        n_examples  = self.data.max_length
        n_iters     = int(np.ceil(float(n_examples)/self.batch_size))

        # Summary
        print("Data size:  %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations: %d" %n_iters)

        #-----------------------Bbox-Prediction---------------------------------
        # Build Attention model
        tf.reset_default_graph()
        pred_bboxs_, _    = self.attn_model.build_test_model()

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Tensorboard summary path
            saver = tf.train.Saver()
            # Load pretrained model
            if self.pretrained_attn_model is not None:
                print("Start testing with pretrained Attention Model..")
                saver.restore(sess,self.pretrained_attn_model)
            else:
                print("Start testing with Attention Model with random weights...")

            pred_bboxes = []
            labl_batchs = []
            for i in range(n_iters):
                image_batch, image_norm_batch, label_batch = next(test_loader)
                # Box Prediction
                feed_dict = {attn_model.images: image_norm_batch,
                             attn_model.drop_prob: 1.0}
                prediction_bboxes = sess.run(pred_bboxs_, feed_dict)
                # Collect
                pred_bboxes.append(prediction_bboxes)
                labl_batchs.append(label_batch)
                if i%self.print_every == 0:
                    print('Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')
        # Close session
        sess.close()

        # Get Valid Predictions boxes
        all_images_crop_rez = []
        for i in range(len(pred_bboxes)):
            vald_prdct_bbxs = valid_pred_bboxs(pred_bboxes=pred_bboxes[i])
            images_crop_rez = crop_and_resize(images=image_batch, bboxes=vald_prdct_bbxs)
            all_images_crop_rez.append(images_crop_rez)

        #------------------------Digit-Classification---------------------------
        # Reset graph
        tf.reset_default_graph()
        # Load Classification Model
        clfy_model = Clfy_Model(image_height=16, image_width=16, mode='test')
        # Predictions
        _, predictions  = clfy_model.build_test_model()
        # Classification score
        check_predictions = tf.equal(predictions, tf.argmax(self.clfy_model.labels, axis=1))
        # To load
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Load pretrained model
            if pretrained_clfy_model is not None:
                print("Start testing with Classification pretrained Model..")
                saver.restore(sess, pretrained_clfy_model)
            else:
                print("Start testing with Classification Model with random weights...")
            # Run predictions
            empty=[]
            lbl_batchs = []
            all_prdcts = []
            for i in range(n_iters):
            for i in range(len(images_crop_rez)):
                input_seq = np.array(images_crop_rez[i])
                try:
                    # Test if any sequence exist
                    input_seq[0]
                    # Predictions
                    feed_dict = {clfy_model.images: input_seq,
                                 clfy_model.drop_prob: 1.0}
                    pred = sess.run(predictions, feed_dict)
                    all_predictions.append(pred)
                except IndexError:
                    all_predictions.append(empty)

                # Collect all outputs
                all_predictions.append(interm_predictions)
                if i%self.print_every == 0:
                    print('Completion..{%d/%d}' % (i, n_iters))
        print('Epoch Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')
        # Close session
        sess.close()

        #----------------------Compute Sequence Accuracy------------------------
        total_acc = 0.0
        total_seq = 0.0
        for t in range(len(labl_batchs)):
            each_lbl_batch = labl_batchs[t]
            each_prd_batch = all_predictions[t]
            for T in range(len(each_lbl_batch)):
                each_labl_seq = each_lbl_batch[T]
                try :
                    each_pred_seq = each_prd_batch[T]
                    seq_accuracy  = 0
                    for tT in range(len(each_labl_seq)):
                        each_labl = each_labl_seq[tT]
                        try:
                            each_pred = each_pred_seq[tT]
                            if each_labl == each_pred:
                                seq_accuracy += 1

                        except IndexError:
                            print('Index Error Occured at: ', t, T, tT)
                        # Get the accuracy of sequence
                        seq_accuracy = seq_accuracy/len(each_labl_seq)
                        # Update sequence
                        total_seq += 1
                        # Update
                        total_acc += seq_accuracy
                except IndexError:
                    print('Index Error Occured at: ',t, T)

        print('Final Sequence accuracy is: %f' % (total_acc/n_examples))
#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset', dataset_dir='test_curated',\
                      dataset_name='test.txt', max_steps=7, mode='Test')
    # Load Attention Model
    attn_model = Attn_Model(dim_feature=[49, 128], dim_hidden=128, n_time_step=7,
                  alpha_c=1.0, image_height=64, image_width=64, mode='test')
    # Load
    clfy_model = Clfy_Model(image_height=16, image_width=16, l2=0.0002, mode='test')
    # Load Inference model
    # testing = Test(data, attn_model, clfy_model, batch_size=16, img_size=(),\
    #                print_every=2000, pretrained_clfy_model='model/lstm1/model-2000',\
    #                pretrained_attn_model='model/lstm1/model-640')
    testing = Test(data, attn_model, clfy_model, img_size=(16,16), batch_size=16,\
                   print_every=2000, pretrained_clfy_model=None, pretrained_attn_model=None)
    # Begin Evaluation
    testing.test()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
