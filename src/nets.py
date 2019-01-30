import tensorflow as tf
slim = tf.contrib.slim

def _batch_norm(x, mode='train', name=None):
    with tf.variable_scope(name + '_batch_norm'):
        tf.layers.batch_normalization(inputs=x,
                                      center=True,
                                      scale=True,
                                      training=(mode=='train'))
    return x

def net(images, mode='train'):
    with tf.variable_scope('CNN'):
        layer_1 = slim.conv2d(images, 64, [5, 5],
                            activation_fn=None,
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=2, scope='layer_1')
        layer_1 = _batch_norm(layer_1, mode=mode, name='layer_1')
        layer_1 = tf.nn.leaky_relu(layer_1, name='relu_layer_1')

        layer_2 = slim.conv2d(layer_1, 128, [3, 3],
                            activation_fn=None,
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=2, scope='layer_2')
        layer_2 = _batch_norm(layer_2, mode=mode, name='layer_2')
        layer_2 = tf.nn.leaky_relu(layer_2, name='relu_layer_2')

        layer_3 = slim.conv2d(layer_2, 128, [3, 3],
                            activation_fn=None,
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=1, scope='layer_3')
        layer_3 = _batch_norm(layer_3, mode=mode, name='layer_3')
        layer_3 = tf.nn.leaky_relu(layer_3, name='relu_layer_3')

        layer_4 = slim.conv2d(layer_3, 128, [3, 3],
                            activation_fn=None,
                            padding='VALID',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=2, scope='layer_4')
        layer_4 = _batch_norm(layer_4, mode=mode, name='layer_4')

        return layer_4

def classification_network(images, dropout, mode='train'):
    with tf.variable_scope('classification_CNN'):
        layer_1 = slim.conv2d(images, 64, [5, 5],
                            activation_fn=None,
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=1, scope='layer_1')
        layer_1 = _batch_norm(layer_1, mode=mode, name='layer_1')
        layer_1 = tf.nn.leaky_relu(layer_1, name='relu_layer_1')
        #print(layer_1.get_shape())

        layer_2 = slim.conv2d(layer_1, 96, [3, 3],
                            activation_fn=None,
                            padding='VALID',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=1, scope='layer_2')
        layer_2 = _batch_norm(layer_2, mode=mode, name='layer_2')
        layer_2 = tf.nn.leaky_relu(layer_2, name='relu_layer_2')
        #print(layer_2.get_shape())

        layer_3 = slim.conv2d(layer_2, 128, [3, 3],
                            activation_fn=None,
                            padding='SAME',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=2, scope='layer_3')
        layer_3 = _batch_norm(layer_3, mode=mode, name='layer_3')
        layer_3 = tf.nn.leaky_relu(layer_3, name='relu_layer_3')
        #print(layer_3.get_shape())

        layer_4 = slim.conv2d(layer_3, 256, [3, 3],
                            activation_fn=None,
                            padding='VALID',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=1, scope='layer_4')
        layer_4 = _batch_norm(layer_4, mode=mode, name='layer_4')
        #print(layer_4.get_shape())

        layer_5 = slim.conv2d(layer_4, 256, [3, 3],
                            activation_fn=None,
                            padding='VALID',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=1, scope='layer_5')
        layer_5 = _batch_norm(layer_5, mode=mode, name='layer_5')
        layer_5 = tf.nn.leaky_relu(layer_5, name='relu_layer_5')
        #print(layer_5.get_shape())

        layer_6 = slim.conv2d(layer_5, 128, [3, 3],
                            activation_fn=None,
                            padding='VALID',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'),
                            stride=2, scope='layer_6')
        layer_6 = _batch_norm(layer_6, mode=mode, name='layer_6')
        layer_6 = tf.nn.leaky_relu(layer_6, name='relu_layer_6')
        #print(layer_6.get_shape())

    with tf.variable_scope('classification_Fully_Connected'):
        layer_6 = slim.flatten(layer_6)
        layer_6 = tf.nn.dropout(layer_6, keep_prob=dropout)
        layer_7 = tf.contrib.layers.fully_connected(layer_6, 128,
                                                    activation_fn=None)
        layer_8 = _batch_norm(layer_7, mode=mode, name='layer_9')
        layer_8 = tf.nn.leaky_relu(layer_8, name='relu_layer_9')
        layer_8 = tf.nn.dropout(layer_8, keep_prob=dropout)
        # Classification layer
        layer_9 = tf.contrib.layers.fully_connected(layer_8, 10,
                                                    activation_fn=None)

    return layer_9
