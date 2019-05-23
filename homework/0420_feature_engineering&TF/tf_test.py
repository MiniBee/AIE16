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
import tensorflow.contrib.slim as slim

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import time
import os

from sklearn.preprocessing import PolynomialFeatures
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
    learn_rate = 5e-1

    file = np.load('./data/homework.npz')
    x = file['X']

    d = file['d']

    label_one_hot = []
    for x1, x2 in x:
        if x1 > 0 and x2 > 0:
            label_one_hot.append([1, 0])
        elif x1 < 0 and x2 < 0:
            label_one_hot.append([1, 0])
        else:
            label_one_hot.append([0, 1])
    label_one_hot = np.array(label_one_hot)

    inputs = tf.placeholder(tf.float32, [batch_size, 2])
    target = tf.placeholder(tf.float32, [batch_size, 2])

    net = tf.layers.dense(inputs, 6, activation=tf.nn.relu)
    net = tf.layers.dense(net, 12, activation=tf.nn.relu)
    net = tf.layers.dense(net, 6, activation=tf.nn.relu)
    y = tf.layers.dense(net, 2, activation=None)

    loss = (y - target) ** 2
    loss = tf.reduce_mean(loss)

    opt = tf.train.GradientDescentOptimizer(learn_rate)
    step = opt.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for itr in range(500):
        idx = np.random.randint(0, 2000, 20)
        inx = x[idx]
        ind = label_one_hot[idx]
        st, ls = sess.run([step, loss], feed_dict={inputs: inx, target: ind})
        if itr % 30 == 0:
            acc = sess.run(accuracy, feed_dict={inputs: x, target: label_one_hot})
            print("step:{}  accuarcy:{}".format(itr, acc))


def graph_layers1():
    files = np.load("./data/homework.npz")
    X = files['X']
    label = files['d']
    label_one_hot = []
    for x1, x2 in X:
        if x1 > 0 and x2 > 0:
            label_one_hot.append([1, 0])
        elif x1 < 0 and x2 < 0:
            label_one_hot.append([1, 0])
        else:
            label_one_hot.append([0, 1])
    label_one_hot = np.array(label_one_hot)

    x = tf.placeholder(tf.float32, [None, 2], name="input_x")
    d = tf.placeholder(tf.float32, [None, 2], name="input_y")
    # 对于sigmoid激活函数而言，效果可能并不理想
    net = slim.fully_connected(x, 4, activation_fn=tf.nn.relu,
                               scope='full1', reuse=False)
    net = slim.fully_connected(net, 4, activation_fn=tf.nn.relu,
                               scope='full4', reuse=False)
    y = slim.fully_connected(net, 2, activation_fn=None,
                             scope='full5', reuse=False)

    loss = tf.reduce_mean(tf.square(y - d))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    gradient = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    train_step = optimizer.apply_gradients(gradient)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for itr in range(500):
        idx = np.random.randint(0, 2000, 20)
        inx = X[idx]
        ind = label_one_hot[idx]
        sess.run(train_step, feed_dict={x: inx, d: ind})
        if itr % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: X, d: label_one_hot})
            print("step:{}  accuarcy:{}".format(itr, acc))


def fe(x):
    x = [list(i) for i in x]
    for i in x:
        i.append(1 if i[0] * i[1] > 0 else 0)
    x = np.array(x)
    return x


def graph_layers_fe():
    batch_size = None
    learn_rate = 5e-1

    file = np.load('./data/homework.npz')
    x = file['X']
    d = file['d']

    label_one_hot = []
    for x1, x2 in x:
        if x1 > 0 and x2 > 0:
            label_one_hot.append([1, 0])
        elif x1 < 0 and x2 < 0:
            label_one_hot.append([1, 0])
        else:
            label_one_hot.append([0, 1])
    label_one_hot = np.array(label_one_hot)

    #
    x = fe(x)

    inputs = tf.placeholder(tf.float32, [batch_size, 3])
    target = tf.placeholder(tf.float32, [batch_size, 2])

    net = tf.layers.dense(inputs, 6, activation=tf.nn.relu)
    net = tf.layers.dense(net, 12, activation=tf.nn.relu)
    net = tf.layers.dense(net, 6, activation=tf.nn.relu)
    y = tf.layers.dense(net, 2, activation=None)

    loss = (y - target) ** 2
    loss = tf.reduce_mean(loss)

    opt = tf.train.GradientDescentOptimizer(learn_rate)
    step = opt.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for itr in range(500):
        idx = np.random.randint(0, 2000, 20)
        inx = x[idx]
        ind = label_one_hot[idx]
        st, ls = sess.run([step, loss], feed_dict={inputs: inx, target: ind})
        if itr % 30 == 0:
            acc = sess.run(accuracy, feed_dict={inputs: x, target: label_one_hot})
            print("step:{}  accuarcy:{}".format(itr, acc))

    x_ = np.array([[0.2, 0.2]])
    x_ = fe(x_)
    print(sess.run(y, feed_dict={inputs: x_}))


def single_layer():
    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
    # batch_xs, batch_ys = mnist.train.next_batch(100)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    net = tf.layers.dense(x, 800, activation=tf.nn.relu)
    net = tf.layers.dense(net, 100, activation=tf.nn.relu)
    y = tf.layers.dense(net, 10, activation=None)

    # 先用softmax将y转换为概率
    p = tf.nn.softmax(y)
    # 在求softmax结果的交叉熵，交叉熵越小，表示距离越小
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


def view_data(file):
    print(file)
    data = file['X']
    y = file['d']
    plt.scatter(data[:, 0], data[:, 1], c=y, s=200, marker='.', edgecolors='k')
    plt.show()


def basic_test():
    pass


if __name__ == '__main__':
    # graph layers
    # graph_layers_fe()
    # file = np.load('./data/homework.npz')
    # view_data(file)
    pass

