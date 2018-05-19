import tensorflow as tf
import numpy as np
import structure

def mnist_demo_train(mnist, params, save_path):
    x = tf.placeholder(tf.float32, [None, params.input_dim])
    y = tf.placeholder(tf.float32, [None, params.output_dim])
    logits = structure.demo_full_nn_mnist(x, params.regularizer)
    a = tf.nn.softmax(logits)
    global_step_count = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        params.learning_rate_base,
        global_step_count,
        params.steps_to_decay,
        params.learning_rate_decay_rate,
        staircase=True
    )
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    if not params.regularizer:
        loss_op = loss_op + tf.add_n(tf.get_collection('losses'))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=global_step_count)
    correct_or_not = tf.equal(tf.arg_max(a, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_or_not, dtype=tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, params.iteration_steps):
            batch_x, batch_y = mnist.train.next_batch(params.batch_size)
            sess.run(train_op, feed_dict={
                x: batch_x,
                y: batch_y
            })
            if step % params.display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={
                    x: batch_x,
                    y: batch_y
                })
                print('Step ' + str(step) + ', Minibatch Loss = ' +
                      '{:.4f}'.format(loss) + ', Training Accuracy= ' +
                      '{:.3f}'.format(acc)
                      )
        print('Optimization Finished!')
        print('Testing Accuracy:',
              sess.run(accuracy, feed_dict={
                  x: mnist.test.images,
                  y: mnist.test.labels
              }))
        saver.save(sess, save_path)

