import tensorflow as tf
import neurons

def demo_full_nn_mnist(x, regularizer=None):
    # input - hidden layer 1
    w1 = neurons.get_weight([784, 20], regularizer)
    b1 = neurons.get_bias([20])
    a1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

    # hidden layer 1 - hidden layer 2
    w2 = neurons.get_weight([20, 20], regularizer)
    b2 = neurons.get_bias([20])
    a2 = tf.nn.relu(tf.add(tf.matmul(a1, w2), b2))

    # hidden layer 2 - output layer
    w3 = neurons.get_weight([20, 10], regularizer)
    b3 = neurons.get_bias([10])
    a3 = tf.add(tf.matmul(a2, w3), b3)

    return a3