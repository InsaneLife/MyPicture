#!/usr/bin/env python
# coding=utf-8
"""
python=3.5.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = 2 * np.power(x_data, 3) + np.power(x_data, 2) + noise

# plot the real data
# plt.scatter(x_data, y_data)
# plt.show()

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer l1
l1 = add_layer(xs, 1, 5, activation_function=tf.nn.relu)
# add hidden layer l2
l2 = add_layer(l1, 5, 10, activation_function=tf.nn.relu)

# output layer l2
prediction = add_layer(l2, 10, 1, activation_function=None)
# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# define Optimizer
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

# number of initialization
init = tf.global_variables_initializer()
# iterations
num_boost = 10000
with tf.Session() as sess:
    sess.run(init)
    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    for i in range(num_boost):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(1)
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
