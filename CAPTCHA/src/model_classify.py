from __future__ import print_function
import tensorflow as tf
import numpy as np
from nets import attn_cnn as net

class Model_Classify(object):
    def __init__(self, image_height=64, image_width=64, l2=0.0002, mode='train'):
        self.l2   = l2
        self.mode = mode
        # Weight Initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer  = tf.constant_initializer(0.0)
        self.trunc_initializer  = tf.initializers.truncated_normal(0.01)
        # Placeholder
        self.images    = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.labels    = tf.placeholder(tf.float32, [None, 62])
        self.drop_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def softmax_cross_entropy(self, labels, logits):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

        return loss

    def _prediction_layer(self, inputs, hidden_dim, reuse=False):
        with tf.variable_scope('prediction_layer', reuse=reuse):
            w = tf.get_variable(shape=[hidden_dim, 62], initializer=self.trunc_initializer, name='weights')
            b = tf.get_variable(shape=[62], initializer=self.const_initializer, name='biases')

            out_logits = tf.matmul(inputs, w) + b

            return out_logits

    def build_model(self):
        logits = classification_network(self.images, dropout=self.drop_prob, mode=self.mode)
        logits = self._prediction_layer(inputs=logits, hidden_dim=1024, reuse=False)
        print('CNN build model sucess!')

        final_loss = 0.0
        batch_size = tf.shape(features)[0]
        # Loss
        interm_loss = self.softmax_cross_entropy(labels=self.labels, logits=logits)
        # Collect loss
        final_loss += tf.reduce_sum(interm_loss)

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
        logits = classification_network(self.images, dropout=self.drop_prob, mode=self.mode)
        logits = self._prediction_layer(inputs=logits, hidden_dim=1024, reuse=False)

        print('CNN build model sucess!')

        return logits
