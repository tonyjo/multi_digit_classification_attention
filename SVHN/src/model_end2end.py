from __future__ import print_function
import tensorflow as tf
import numpy as np
from nets import attn_cnn as net

class Model(object):
    def __init__(self, dim_feature=[49, 256], dim_hidden=128, n_time_step=3,
                 alpha_c=0.0, image_height=64, image_width=64, l2=0.0002, mode='train'):
        self.L  = dim_feature[0]
        self.D  = dim_feature[1]
        self.H  = dim_hidden
        self.T  = n_time_step
        self.l2 = l2
        self.mode    = mode
        self.alpha_c = alpha_c
        self.H_attn, self.W_attn = int(np.sqrt(self.L)), int(np.sqrt(self.L))
        # Weight Initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer  = tf.constant_initializer(0.0)
        self.trunc_initializer  = tf.initializers.truncated_normal(0.01)
        # Placeholders
        self.images    = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.labels    = tf.placeholder(tf.float32, [None, 10])
        self.bboxes    = tf.placeholder(tf.float32,   [None, self.T, 4])
        self.gnd_attn  = tf.placeholder(tf.float32,   [None, self.T, self.L])
        self.drop_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def _mean_squared_error(self, grd_bboxes, pred_bboxes):
        loss = tf.losses.mean_squared_error(labels=grd_bboxes, predictions=pred_bboxes)

        return loss

    def _softmax_cross_entropy(self, labels, logits):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

        return loss

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)
            w_h = tf.get_variable(shape=[self.D, self.H], initializer=self.weight_initializer, name='weights')
            b_h = tf.get_variable(shape=[self.H], initializer=self.const_initializer, name='biases')
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)
            w_c = tf.get_variable(shape=[self.D, self.H], initializer=self.weight_initializer, name='w_weights')
            b_c = tf.get_variable(shape=[self.H], initializer=self.const_initializer, name='b_biases')
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)

            return c, h

    def _attention_layer(self, features, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable(shape=[self.H, self.D], initializer=self.weight_initializer, name='weights')
            b = tf.get_variable(shape=[self.D], initializer=self.const_initializer, name='biases')
            w_att = tf.get_variable(shape=[self.D, 2], initializer=self.weight_initializer, name='w_weights')
            h_att   = tf.nn.relu(features + tf.expand_dims(tf.matmul(h, w), 1) + b) # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L, 2]) # (N, L, 2)
            alpha   = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context') #(N, D)

            return context, alpha # out_att # normalized

    def _interm_prediction_layer(self, name_scope, inputs, outputs, reuse=False):
        batch_size = tf.shape(inputs)[0]
        # Average Pool
        avg_inputs = tf.nn.avg_pool(value=inputs, ksize=[1, 3, 3, 1], stides=[1, 2, 2, 1])
        # Flatten
        fln_inputs = tf.reshape(avg_inputs, [batch_size, -1])
        # Shape of input
        B, H = fln_inputs.get_shape().as_list()

        with tf.variable_scope(name_scope, reuse=reuse):
            w = tf.get_variable(shape=[H, output], initializer=self.trunc_initializer, name='weights')
            b = tf.get_variable(shape=[output], initializer=self.const_initializer, name='biases')

            out_logits = tf.matmul(fln_inputs, w) + b

            return out_logits

    def build_model(self):
        features   = net(self.images, mode=self.mode)
        features   = tf.reshape(features, [-1, self.L, self.D])
        print('CNN build model sucess!')
        bboxes     = self.bboxes
        gnd_attn   = self.gnd_attn
        batch_size = tf.shape(features)[0]
        final_loss = 0.0
        alpha_list = []
        lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        lstm_cell  = tf.contrib.rnn.DropoutWrapper(lstm_cell,\
                                                   input_keep_prob=self.drop_prob)
        # Get initial state of LSTM
        c, h = self._get_initial_lstm(features=features)
        # Loop for t steps
        for t in range(self.T):
            # Attention
            context, alpha = self._attention_layer(features, h, reuse=(t!=0))
            alpha_list.append(alpha)
            # LSTM step
            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=context, state=[c, h])
            # BBox Prediction
            bbox_pred = self._prediction_layer(name_scope='bbox_pred_layer', inputs=h,\
                                               output=4, H=self.H, reuse=(t!=0))
            # Digit prediction
            digit_pred = self._interm_prediction_layer(name_scope='interm_digit_pred',\
                                               inputs=context, outputs=1024, reuse=(t!=0))
            digit_pred = self._prediction_layer(name_scope='digit_pred_layer', inputs=digit_pred,\
                                                output=10, H=1024, reuse=(t!=0))
            # Loss
            interm_loss_digit = self._softmax_cross_entropy(labels=self.labels, logits=logits)
            # Loss at each time step
            interm_loss_bbox  = self._mean_squared_error(grd_bboxes=self.bboxes[:, t, :], pred_bboxes=logits)
            # Collect loss
            final_loss += tf.reduce_sum(interm_loss_digit) + tf.reduce_sum(interm_loss_digit)

        if self.alpha_c > 0:
            ## KL-loss
            alpha_loss = 0.0
            for T in range(self.T):
                pred_alpha = alpha_list[T] # (N, L)
                grnd_alpha = self.gnd_attn[:, T, :] # (N, L)
                eror_alpha =  grnd_alpha * tf.log(grnd_alpha/(pred_alpha + 0.0001) + 1e-8)  # Avoid NaN
                alpha_loss += tf.reduce_sum(eror_alpha) # (1)
            # Weight alpha loss
            alpha_reg = self.alpha_c * alpha_loss
            # Add alpha loss to
            final_loss += alpha_reg
        else:
            print('No Attention Regularization!')

        if self.l2 > 0:
            print('L2 regularization:')
            for var in tf.trainable_variables():
                tf_var = var.name
                if tf_var[-8:-2] != 'biases' and tf_var[-6:-2] != 'bias':
                    print(tf_var)
                    final_loss = final_loss + (self.l2 * tf.nn.l2_loss(var))

        return final_loss/tf.to_float(batch_size)

    def build_test_model(self):
        features   = net(self.images, mode=self.mode)
        features   = tf.reshape(features, [-1, self.L, self.D])
        print('CNN build model sucess!')
        alpha_list = []
        pred_bboxs = []
        pred_digit = []
        batch_size = tf.shape(features)[0]
        lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        lstm_cell  = tf.contrib.rnn.DropoutWrapper(lstm_cell,\
                                                   input_keep_prob=self.drop_prob)
        # Get initial state of LSTM
        c, h = self._get_initial_lstm(features=features)
        # Loop for t steps
        for t in range(self.T):
            # Attention
            context, alpha = self._attention_layer(features, h, reuse=(t!=0))
            # LSTM step
            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=context, state=[c, h])
            # BBox Prediction
            bbox_pred = self._prediction_layer(name_scope='bbox_pred_layer', inputs=h,\
                                               output=4, H=self.H, reuse=(t!=0))
            # Digit prediction
            digit_pred = self._interm_prediction_layer(name_scope='interm_pred_layer',\
                                               inputs=context, outputs=1024, reuse=(t!=0))
            digit_pred = self._prediction_layer(name_scope='digit_pred_layer', inputs=digit_pred,\
                                                output=10, H=1024, reuse=(t!=0))
            # Collect
            alpha_list.append(alpha)
            pred_bboxs.append(pred_bboxs)
            pred_digit.append(digit_pred)

        return pred_bboxs, pred_digit, alpha_list
