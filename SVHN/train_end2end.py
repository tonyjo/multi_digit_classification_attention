from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.data_loader_end2end import dataLoader
from src.model_end2end import Model

class Train(object):
    def __init__(self, model, data, val_data, **kwargs):
        self.model         = model
        self.data          = data
        self.val_data      = val_data
        self.max_steps     = model.T
        self.check_val     = kwargs.pop('check_val', 10)
        self.n_epochs      = kwargs.pop('n_epochs', 20)
        self.batch_size    = kwargs.pop('batch_size', 64)
        self.val_bth_size  = kwargs.pop('val_batch_size', 1)
        self.update_rule   = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.0001)
        self.valid_freq    = kwargs.pop('valid_freq', 10)
        self.print_every   = kwargs.pop('print_every', 100)
        self.save_every    = kwargs.pop('save_every', 1)
        self.log_path      = kwargs.pop('log_path', './log/')
        self.model_path    = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

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

    def bb_intersection_over_union(self, boxA, boxB):
        """
        Args:
            bboxes1: shape (total_bboxes1, 4)
                with p1=(x1, y1, x2, y2) point order.
            bboxes2: shape (total_bboxes2, 4)
                with p2=(x1, y1, x2, y2) point order.
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

    def IOU(self, grnd_bboxes, pred_bboxes):
        count = 0.0
        n_iters = len(pred_bboxes)
        final_IOU_score = 0.0
        for t in range(n_iters):
            sample_seq = 0.0
            sample_iou = 0.0
            for T in range(self.max_steps):
                ## Ground
                sample_left_grd  = grnd_bboxes[t][0][T][0]
                sample_top_grd   = grnd_bboxes[t][0][T][1]
                sample_width_grd = grnd_bboxes[t][0][T][2]
                sample_heigt_grd = grnd_bboxes[t][0][T][3]

                ## Predicted
                sample_left  = int(pred_bboxes[t][T][0][0])
                sample_top   = int(pred_bboxes[t][T][0][1])
                sample_width = int(pred_bboxes[t][T][0][2])
                sample_heigt = int(pred_bboxes[t][T][0][3])

                vld_bbox_prd = self.bbox_threshold(left=sample_left, top=sample_top,\
                                        width=sample_width, height=sample_heigt)

                vld_bbox_grd = self.bbox_threshold(left=sample_left_grd, top=sample_top_grd,\
                                        width=sample_width_grd, height=sample_heigt_grd)

                if vld_bbox_grd:
                    sample_seq += 1
                    if vld_bbox_prd:
                        # Predicted
                        sample_right = sample_left + sample_width
                        sample_down  = sample_top  + sample_heigt
                        # Ground truth
                        sample_right_grd = sample_left_grd + sample_width_grd
                        sample_down_grd  = sample_top_grd  + sample_heigt_grd

                        boxA = (sample_left, sample_top, sample_right, sample_down)
                        boxB = (sample_left_grd, sample_top_grd, sample_right_grd, sample_down_grd)

                        sample_iou += self.bb_intersection_over_union(boxA, boxB)
            # Print Every
            if t%2000 == 0:
                print(' Validation IOU Completion..{%d/%d}' % (t, n_iters))
            final_IOU_score += sample_iou/sample_seq
            count += 1.0
        #-----------------------------------------------------------------------
        print('Validation IOU Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')
        print('Final Sequence IOU score = ', np.ceil((final_IOU_score/count)*100), '%')
        #-----------------------------------------------------------------------

    def pred_accuracy(self, grnd_labels, pred_cpthas):
        count = 0.0
        n_iters = len(pred_cpthas)
        final_seq_acc_prd = 0.0
        # Run predictions
        for t in range(n_iters):
            # Get image
            sample_cnt = 0
            sample_seq = len(grnd_labels[t][0])
            sample_acc_prd = 0.0
            # Loop through steps
            for T in range(self.max_steps):
                # Label
                sample_label = np.argmax(grnd_labels[t][0][T])
                # Predicted
                pred_score   = np.argmax(pred_cpthas[t][T][0])
                # Check sample
                if sample_label != 62 or sample_label != 63:
                    if sample_cnt < sample_seq:
                        # Predictions
                        if int(pred_score) == int(sample_label):
                            sample_acc_prd += 1.0
                        # Update
                        sample_cnt += 1
            # Increment Sequence Accuracy
            final_seq_acc_prd += sample_acc_prd/sample_seq
            # Progress
            if t%2000 == 0:
                print('Validation Prediction Completion..{%d/%d}' % (t, n_iters))
            count += 1.0
        #-----------------------------------------------------------------------
        print('Validation Prediction Completion..{%d/%d}' % (n_iters, n_iters))
        print('Completed!')
        print('Validation Sequence Classification Accuracy: ',\
                              np.round((final_seq_acc_prd/count), 4) * 100, '%')
        #-----------------------------------------------------------------------

    def train(self):
        # Train Data Loader
        train_loader = self.data.gen_data_batch(self.batch_size)
        valid_loader = self.val_data.gen_data_batch(self.val_bth_size)

        # Train dataset
        n_examples = self.data.max_length
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        # Validation dataset
        n_examples_val = self.val_data.max_length
        valid_n_iters  = int(np.ceil(float(n_examples_val)/self.val_bth_size))

        # Build model with loss
        loss, pred_bboxs_, pred_cptha_, = self.model.build_model()

        # Global step
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Train op
        with tf.name_scope('optimizer'):
            decay_l_rate = tf.train.exponential_decay(self.learning_rate, global_step,\
                                                       50000, 0.9, staircase=True)
            incr_glbl_stp = tf.assign(global_step, global_step+1)
            optimizer = self.optimizer(learning_rate=decay_l_rate)
            grads     = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)
        # Summary op
        tf.summary.scalar('batch_loss', loss)
        summary_op = tf.summary.merge_all()
        # Show steps
        print("The number of epoch: %d" %self.n_epochs)
        print("Data size: %d"  %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations per epoch: %d"   %n_iters_per_epoch)
        print("Validation interations: %d" %valid_n_iters)

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Tensorboard summary path
            summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(max_to_keep=4)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            for e in range(self.n_epochs):
                curr_loss = 0
                start_t   = time.time()
                for i in range(n_iters_per_epoch):
                    image_batch, grd_labels_batch, grd_bboxes_batch, grd_attn_batch = next(train_loader)
                    feed_dict = {self.model.images: image_batch,
                                 self.model.labels: grd_labels_batch,
                                 self.model.bboxes: grd_bboxes_batch,
                                 self.model.gnd_attn: grd_attn_batch,
                                 self.model.is_train: True,
                                 self.model.drop_prob: 0.5}

                    _, l, _ = sess.run([train_op, loss, incr_glbl_stp], feed_dict)
                    curr_loss += l

                    if i%self.print_every == 0:
                        print('Epoch Completion..{%d/%d}' % (i, n_iters_per_epoch))
                    write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)
                # Check validation
                if e%self.check_val == 0:
                    pred_cpthas = []
                    pred_bboxes = []
                    grnd_bboxes = []
                    grnd_labels = []
                    for t in range(valid_n_iters):
                        image_val_batch, grd_val_lables_batch, grd_val_bboxes_batch = next(valid_loader)
                        feed_dict = {self.model.images: image_val_batch,
                                     self.model.is_train: False,
                                     self.model.drop_prob: 1.0}
                        prediction_bbox, prediction_cptha = sess.run([pred_bboxs_, pred_cptha_], feed_dict)
                        # Collect
                        pred_bboxes.append(prediction_bbox)
                        pred_cpthas.append(prediction_cptha)
                        grnd_bboxes.append(grd_val_bboxes_batch)
                        grnd_labels.append(grd_val_lables_batch)
                        # Print every
                        if t%1000 == 0:
                            print('Inference Completion..{%d/%d}' % (t, valid_n_iters))
                    #-----------------------------------------------------
                    print('Inference Completion..{%d/%d}' % (valid_n_iters, valid_n_iters))
                    print('Completed!')
                    # IOU
                    self.IOU(grnd_bboxes=grnd_bboxes, pred_bboxes=pred_bboxes)
                    # Prediction Accuracy
                    self.pred_accuracy(grnd_labels=grnd_labels, pred_cpthas=pred_cpthas)
                #---------------------------------------------------------------
                print("Previous epoch loss: ", prev_loss)
                print("Current epoch loss: ", curr_loss)
                print("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                #---------------------------------------------------------------
                # Save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print("model-%s saved." %(e+1))
        # Close session
        sess.close()
#-------------------------------------------------------------------------------
def main():
    # Load train/val dataset
    data = dataLoader(directory='./dataset', dataset_dir='train_cropped',\
                      dataset_name='train.txt', max_steps=6, image_width=64,\
                      image_height=64, grd_attn=True, mode='Train')
    val_data = dataLoader(directory='./dataset', dataset_dir='train_cropped',\
                      dataset_name='val.txt', max_steps=6, image_width=64,\
                      image_height=64, grd_attn=False, mode='Valid')
    # Load Model
    model = Model(dim_feature=[196, 128], dim_hidden=128, n_time_step=6,
                  alpha_c=5.0, image_height=64, image_width=64, mode='train')
    # Load Trainer
    trainer = Train(model, data, val_data=val_data, n_epochs=1000, batch_size=64, val_batch_size=1,
                    update_rule='adam', learning_rate=0.0001, print_every=100, valid_freq=10,
                    save_every=5, pretrained_model=None, model_path='model/lstm3/', log_path='log3/')
    # Begin Training
    trainer.train()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
