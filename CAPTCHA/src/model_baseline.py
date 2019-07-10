from __future__ import print_function
import tensorflow as tf
import numpy as np
from nets import baseline_network

class Model_Baseline(object):
    def __init__(self, image_height=64, image_width=64, l2=0.0002, mode='train'):
        self.l2   = l2
        self.mode = mode
        # Weight Initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer  = tf.constant_initializer(0.0)
        self.trunc_initializer  = tf.initializers.truncated_normal(0.01)
        # Placeholder
        self.images     = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.labels_ZL  = tf.placeholder(tf.float32, [None, 6])
        self.labels_ZS1 = tf.placeholder(tf.float32, [None, 62])
        self.labels_ZS2 = tf.placeholder(tf.float32, [None, 62])
        self.labels_ZS3 = tf.placeholder(tf.float32, [None, 62])
        self.labels_ZS4 = tf.placeholder(tf.float32, [None, 62])
        self.labels_ZS5 = tf.placeholder(tf.float32, [None, 62])
        self.labels_ZS6 = tf.placeholder(tf.float32, [None, 62])
        self.drop_prob  = tf.placeholder(tf.float32, name='dropout_prob')

    def softmax_cross_entropy(self, labels, logits):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

        return loss

    def _prediction_layer(self, scope, inputs, hidden_dim, output_dim, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            w = tf.get_variable(shape=[hidden_dim, output_dim], initializer=self.trunc_initializer, name='weights')
            b = tf.get_variable(shape=[output_dim], initializer=self.const_initializer, name='biases')

            out_logits = tf.matmul(inputs, w) + b

            return out_logits

    def build_model(self):
        # Load baseline network
        logits = baseline_network(self.images, dropout=self.drop_prob, mode=self.mode)
        print('Baseline build model sucess!')
        _, l_h = logits.get_shape().as_list()
        # Add prediction networks
        # Z_L
        Z_L  = self._prediction_layer(scope="Z_L",  inputs=logits, hidden_dim=l_h, output_dim=6, reuse=False)
        # Z_S1
        Z_S1 = self._prediction_layer(scope="Z_S1", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S2
        Z_S2 = self._prediction_layer(scope="Z_S2", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S3
        Z_S3 = self._prediction_layer(scope="Z_S3", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S4
        Z_S4 = self._prediction_layer(scope="Z_S4", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S5
        Z_S5 = self._prediction_layer(scope="Z_S5", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S6
        Z_S6 = self._prediction_layer(scope="Z_S6", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)

        # Loss
        final_loss = 0.0
        batch_size = tf.shape(logits)[0]
        Z_L_loss = self.softmax_cross_entropy(labels=self.labels_Z_L, logits=Z_L)
        ZS1_loss = self.softmax_cross_entropy(labels=self.labels_ZS1, logits=Z_S1)
        ZS2_loss = self.softmax_cross_entropy(labels=self.labels_ZS2, logits=Z_S2)
        ZS3_loss = self.softmax_cross_entropy(labels=self.labels_ZS3, logits=Z_S3)
        ZS4_loss = self.softmax_cross_entropy(labels=self.labels_ZS4, logits=Z_S4)
        ZS5_loss = self.softmax_cross_entropy(labels=self.labels_ZS5, logits=Z_S5)
        ZS6_loss = self.softmax_cross_entropy(labels=self.labels_ZS6, logits=Z_S6)
        # Collect loss
        final_loss = Z_L_loss + ZS1_loss + ZS2_loss + ZS3_loss + ZS4_loss + ZS5_loss + ZS6_loss

        # L2-regularization
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
        # Load baseline network
        logits = baseline_network(self.images, dropout=self.drop_prob, mode=self.mode)
        print('Baseline build model sucess!')
        _, l_h = logits.get_shape().as_list()
        # Add prediction networks
        # Z_L
        Z_L  = self._prediction_layer(scope="Z_L",  inputs=logits, hidden_dim=l_h, output_dim=6, reuse=False)
        # Z_S1
        Z_S1 = self._prediction_layer(scope="Z_S1", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S2
        Z_S2 = self._prediction_layer(scope="Z_S2", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S3
        Z_S3 = self._prediction_layer(scope="Z_S3", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S4
        Z_S4 = self._prediction_layer(scope="Z_S4", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S5
        Z_S5 = self._prediction_layer(scope="Z_S5", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)
        # Z_S6
        Z_S6 = self._prediction_layer(scope="Z_S6", inputs=logits, hidden_dim=l_h, output_dim=62, reuse=False)

        return Z_L, Z_S1, Z_S2, Z_S3, Z_S4, Z_S5, Z_S6
