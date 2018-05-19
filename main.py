import tensorflow as tf
from collections import namedtuple
import train
# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

params_struct = namedtuple('params', [
    'learning_rate_base',
    'learning_rate_decay_rate',
    'steps_to_decay',
    'iteration_steps',
    'batch_size',
    'display_step',
    'input_dim',
    'hidden_1_dim',
    'hidden_2_dim',
    'output_dim',
    'regularizer'
])

params = params_struct(
    learning_rate_base=0.01,
    learning_rate_decay_rate=0.99,
    steps_to_decay=500,
    iteration_steps=100000,
    batch_size=128,
    display_step=500,
    input_dim=784,
    hidden_1_dim=20,
    hidden_2_dim=20,
    output_dim=10,
    regularizer=0.05
)

if __name__ == '__main__':
    train.mnist_demo_train(mnist, params, save_path='./model/model.ckpt')