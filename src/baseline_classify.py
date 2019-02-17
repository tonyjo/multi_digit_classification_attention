from __future__ import print_function
import tensorflow as tf
import numpy as np
from nets import base_classification_network as base_classification_network

class Model(object):
    def __init__(self, image_height=16, image_width=16, l2=0.0002, mode='train'):
        self.l2 = l2
        self.mode = mode
        # Placeholder
        self.images = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.drop_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def build_model(self):
        logits = base_classification_network(self.images, dropout=self.drop_prob, mode=self.mode)
        print('Classification build model sucess!')

        batch_size = tf.shape(logits)[0]
        # Loss at each time step
        final_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,\
                                                             logits=logits)
        # Collect loss
        final_loss = tf.reduce_sum(final_loss)

        if self.l2 > 0:
            print('L2 regularization:')
            for var in tf.trainable_variables():
                tf_var = var.name
                if tf_var[-8:-2] != 'biases' and tf_var[-6:-2] != 'bias':
                    print(tf_var)
                    final_loss = final_loss + (self.l2 * tf.nn.l2_loss(var))
        print('...............................................................')

        return final_loss/tf.to_float(batch_size)

    def build_test_model(self):
        logits = base_classification_network(self.images, dropout=self.drop_prob, mode=self.mode)
        logits_norm = tf.nn.softmax(logits=logits)

        print('Classification build model sucess!')

        prediction = tf.argmax(input=logits_norm, axis=1, name='pred_argmax')

        return logits_norm, prediction
