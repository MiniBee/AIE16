#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: tf_test.py
# @time: 2019/4/28 上午11:33
# @desc:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def session_test():
    # test1
    hello = tf.constant('hello TensorFlow')
    with tf.Session() as sess:
        print(sess.run(hello))

    # test2
    a = tf.constant(3)
    b = tf.constant(4)
    with tf.Session() as sess:
        print('a + b = {0}'.format(sess.run(a + b)))
        print('a * b = {0}'.format(sess.run(a * b)))

    # 交互式session
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])
    x.initializer.run()
    sub = tf.subtract(x, a)
    print(sub.eval())
    sess.close()

    # Session注入机制
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add = a + b
    product = a * b
    print(a, b, add, product)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('a + b = {0}'.format(sess.run(add, feed_dict={a: 3, b: 4})))
        print('a * b = {0}'.format(sess.run(product, feed_dict={a: 3, b: 4})))
        print(sess.run([add, product], feed_dict={a: 3, b: 4}))


def graph_layers():
    batch_size = None
    learn_rate = 1e-3
    inputs = tf.placeholder(tf.float32, [batch_size, 2])
    target = tf.placeholder(tf.float32, [batch_size, 1])

    h1 = tf.layers.dense(inputs, 6, activation=tf.nn.relu)
    a = np.linspace(-2, 4, 100)
    w = tf.get_variable("w", [1, 1])
    w2 = tf.get_variable("w2", [1, 1])
    b = tf.get_variable("b", [1])
    y = w[0, 0] * a + b[0] + w2[0, 0] * a ** 2

    loss = (y-target) ** 2
    loss = tf.reduce_mean(loss)

    opt = tf.train.GradientDescentOptimizer(learn_rate)
    step = opt.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    file = np.load('./data/homework.npz')
    x = file['X']
    d = file['d']
    for itr in range(500):
        idx = np.random.randint(0, 1000, 32)
        inx = x[idx]
        ind = d[idx]
        st, ls = sess.run([step, loss], feed_dict={inputs: inx, target: ind})
        print(itr, ls)
    w, w2, b = sess.run([w, w2, b])

    a = np.linspace(-2, 4, 100)
    y = w[0, 0] * a + b[0] + w2[0, 0] * a ** 2
    plt.scatter(x[:, 0], d[:, 0])
    plt.plot(a, y, lw=3, color="#000000")
    plt.show()


def single_layer():
    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
    # batch_xs, batch_ys = mnist.train.next_batch(100)

    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    y = tf.layers.dense(x, 10, activation=None)
    p = tf.nn.softmax(y)

    loss = tf.reduce_sum(- label * tf.log(p), axis=1)
    loss = tf.reduce_mean(loss)

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    corrent_predication = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(corrent_predication, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./data/mnist-logdir', sess.graph)
    for itr in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
        if itr % 10 == 0:
            print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))
    sess.close()


if __name__ == '__main__':
    # error
    # graph_layers()

    # single layer
    single_layer()
