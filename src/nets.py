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
