#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: mnist_tf.py
# @time: 2019/4/29 下午4:01
# @desc:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# step:   980  accuracy: 0.8082
# step:   990  accuracy: 0.8096
def single_layer(mnist):
    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.sigmoid(tf.matmul(x, W) + b)

    loss = tf.reduce_mean(tf.square(y - label))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('mnist-logdir', sess.graph)
    for itr in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
        if itr % 10 == 0:
            print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))


# step:   980  accuracy: 0.8698
# step:   990  accuracy: 0.8707
def single_layer_crossentropy(mnist):
    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    logits = tf.matmul(x, W) + b
    prob = tf.nn.softmax(logits)

    loss = tf.reduce_mean(- label * tf.log(prob))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for itr in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
        if itr % 10 == 0:
            print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))


# step:   980  accuracy: 0.7398
# step:   990  accuracy: 0.7428
def single_layer_opt(mnist):
    def full_layer(input_tensor, out_dim, name='full'):
        with tf.variable_scope(name):
            shape = input_tensor.get_shape().as_list()
            W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
            out = tf.matmul(input_tensor, W) + b
        return tf.nn.sigmoid(out)

    def model(net, out_dim):
        net = full_layer(net, out_dim, 'full_layer1')
        return net

    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        label = tf.placeholder(tf.float32, [None, 10])

    y = model(x, 10)

    loss = tf.reduce_mean(tf.square(y - label))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for itr in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
        if itr % 10 == 0:
            print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                             label: mnist.test.labels}))


def single_layer_save(mnist):
    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # sigmoid
    # y = tf.nn.sigmoid(tf.matmul(x, W) + b)
    # softmax cross_entropy
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = tf.reduce_sum(-label * tf.log(y), axis=1)
    loss = tf.reduce_mean(loss)

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    xs, ys = mnist.train.next_batch(100)
    for itr in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
        if itr % 10 == 0:
            print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))
            # saver.save(sess, os.path.join(os.getcwd(), 'model', 'mnist'), global_step=itr)

    sess.run(train_step, feed_dict={x: xs, label: ys})
    print("step:%6d  accuracy:" % sess.run(accuracy, feed_dict={x: xs, label: ys}))
    print(ys)
    print(sess.run(y, feed_dict={x: xs, label: ys}))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
    single_layer_save(mnist)



