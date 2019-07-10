from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.data_loader_baseline import dataLoader
from src.model_baseline import Model_Baseline

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
        loss, Z_L, Z_S1, Z_S2, Z_S3, Z_S4, Z_S5, Z_S6 = self.model.build_model()

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
                    image_batch, grd_labels_batch, grd_mxstep_batch = next(train_loader)
                    feed_dict = {self.model.images: image_batch,
                                 self.model.labels_Z_L: grd_mxstep_batch,
                                 self.model.labels_ZS1: grd_labels_batch[:, 0, :],
                                 self.model.labels_ZS2: grd_labels_batch[:, 1, :],
                                 self.model.labels_ZS3: grd_labels_batch[:, 2, :],
                                 self.model.labels_ZS4: grd_labels_batch[:, 3, :],
                                 self.model.labels_ZS5: grd_labels_batch[:, 4, :],
                                 self.model.labels_ZS6: grd_labels_batch[:, 5, :],
                                 self.model.is_train: True,
                                 self.model.drop_prob: 0.5}
                    # Run Optim
                    _, l, _ = sess.run([train_op, loss, incr_glbl_stp], feed_dict)
                    curr_loss += l

                    if i%self.print_every == 0:
                        print('Epoch Completion..{%d/%d}' % (i, n_iters_per_epoch))
                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)
                # Check validation
                if e%self.check_val == 0:
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
                    print('Validation Sequence Classification Accuracy: ',\
                              np.round((final_seq_acc_prd/n_iters), 4) * 100, '%')
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
    data = dataLoader(directory='./dataset/captcha', dataset_dir='train',\
                      dataset_name='train.txt', max_steps=6, image_width=200,\
                      image_height=64, grd_attn=True, mode='Train')
    val_data = dataLoader(directory='./dataset/captcha', dataset_dir='val',\
                      dataset_name='val.txt', max_steps=6, image_width=200,\
                      image_height=64, grd_attn=False, mode='Valid')
    # Load Model
    model = Model_Baseline(image_height=64, image_width=200, mode='train')
    # Load Trainer
    trainer = Train(model, data, val_data=val_data, n_epochs=1000, batch_size=64, val_batch_size=1,
                    update_rule='adam', learning_rate=0.0001, print_every=100, valid_freq=10,
                    save_every=5, pretrained_model=None, model_path='model/lstm2/', log_path='log2/')
    # Begin Training
    trainer.train()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
