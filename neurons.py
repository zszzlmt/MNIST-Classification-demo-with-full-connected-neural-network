import tensorflow as tf

def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b
