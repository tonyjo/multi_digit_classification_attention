from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import tensorflow as tf
from src.model_classify import Model_Classify
from src.data_loader_classify import dataLoader

class Train(object):
    def __init__(self, model, data, val_data, **kwargs):
        self.model         = model
        self.data          = data
        self.val_data      = val_data
        self.n_epochs      = kwargs.pop('n_epochs', 20)
        self.batch_size    = kwargs.pop('batch_size', 64)
        self.update_rule   = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.0001)
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

    def train(self):
        # Train Data Loader
        train_loader = self.data.gen_data_batch(self.batch_size)

        # Train dataset
        n_examples = self.data.max_length
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))

        # Build model with loss
        loss = self.model.build_model()

        # Global step
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Train op
        with tf.name_scope('optimizer'):
            decay_l_rate = tf.train.exponential_decay(self.learning_rate, global_step,\
                                                       28125, 0.1, staircase=True)
            incr_glbl_stp = tf.assign(global_step, global_step+1)
            optimizer = self.optimizer(learning_rate=decay_l_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # Summary op
        tf.summary.scalar('batch_loss', loss)

        summary_op = tf.summary.merge_all()

        print("The number of epoch: %d" %self.n_epochs)
        print("Data size: %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations per epoch: %d" %n_iters_per_epoch)

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # Intialize the training graph
            sess.run(tf.global_variables_initializer())
            # Tensorboard summary path
            summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(max_to_keep=5)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            for e in range(self.n_epochs):
                curr_loss = 0
                start_t   = time.time()
                for i in range(n_iters_per_epoch):
                    image_batch, label_batch = next(train_loader)
                    feed_dict = {self.model.images: image_batch,
                                 self.model.labels: label_batch,
                                 self.model.drop_prob: 0.5}

                    _, l, _ = sess.run([train_op, loss, incr_glbl_stp], feed_dict)
                    curr_loss += l

                    if i%self.print_every == 0:
                        print('Epoch Completion..{%d/%d}' % (i, n_iters_per_epoch))

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                print("Previous epoch loss: ", prev_loss)
                print("Current epoch loss: ", curr_loss)
                print("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss

                # Save model's parameters
                if (e) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print("model-%s saved." %(e+1))
        # Close session
        sess.close()
#-------------------------------------------------------------------------------
def main():
    # Load train dataset
    data = dataLoader(directory='./dataset/captcha', dataset_dir='train',\
                      dataset_name='train.txt', max_steps=6, image_width=200,\
                      image_height=64, image_patch_width=54, image_patch_height=54,\
                      grd_attn=False, mode='Train')

    # Load Model
    model = Model_Classify(image_height=54, image_width=54, l2=0.0002, mode=True)

    # Load Trainer
    trainer = Train(model, data, val_data=None, n_epochs=400, batch_size=128,
                    update_rule='adam', learning_rate=0.0001, print_every=500, save_every=5,
                    pretrained_model=None, model_path='model/clsfy1/', log_path='log_clsfy1/')

    # Begin Training
    trainer.train()
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
