#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: iris_tf.py
# @time: 2019/4/28 上午10:24
# @desc:

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def norm_x(x):
    x = x - np.mean(x, axis=0)
    x = x / np.max(x, axis=0)
    return x


if __name__ == '__main__':
    data = pd.read_csv('./data/iris.data.csv')

    iris_x = data.values[:, :-1]
    # 归一化
    iris_x = norm_x(iris_x)

    c_name = set(data['name'].values)
    iris_y = np.zeros([len(data['name'].values), len(c_name)])

    # one hot Y
    len_of_data = []
    for idx, itr_name in enumerate(c_name):
        len_of_data.append(len([iris_y[data.name.values==itr_name]]))
        iris_y[data.name.values==itr_name, idx] = 1
    # print(len(len_of_data))
    # print(iris_y)

    x = tf.placeholder(tf.float32, [None, 4], name='input_x')
    label = tf.placeholder(tf.float32, [None, 3], name='input_y')

    net = slim.fully_connected(x, 4, activation_fn=tf.nn.relu, scope='full1', reuse=False)
    net = slim.fully_connected(net, 4, activation_fn=tf.nn.relu, scope='full2', reuse=False)
    y = slim.fully_connected(net, 3, activation_fn=tf.nn.sigmoid, scope='full3', reuse=False)

    loss = tf.reduce_mean(tf.square(y - label))

    correct_predication = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predication, tf.float32))

    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for itr in range(600):
        sess.run(train_step, feed_dict={x: iris_x[:100], label: iris_y[:100]})
        if itr % 30 == 0:
            acc = sess.run(accuracy, feed_dict={x: iris_x[:100], label: iris_y[:100]})
            print("step:{:6d}  accuracy:{:.3f}".format(itr, acc))
    

