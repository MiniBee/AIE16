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


def iris_tf1(data):
    iris_x = data.values[:, :-1]
    # 归一化
    iris_x = norm_x(iris_x)

    c_name = set(data['name'].values)
    iris_y = np.zeros([len(data['name'].values), len(c_name)])

    # one hot Y
    len_of_data = []
    for idx, itr_name in enumerate(c_name):
        len_of_data.append(len([iris_y[data.name.values == itr_name]]))
        iris_y[data.name.values == itr_name, idx] = 1
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


def iris_tf2(data):
    c_name = set(data.name.values)
    iris_data = data.values[:, :-1]
    iris_data = norm_x(iris_data)
    iris_label = np.zeros([len(data.name.values), len(c_name)])

    train_data = []
    train_data_label = []
    test_data = []
    test_data_label = []

    for idx, itr_name in enumerate(c_name):
        data_t = iris_data[data.name.values==itr_name, :]
        label_t = np.zeros([len(data_t), len(c_name)])
        label_t[:, idx] = 1
        train_data.append(data_t[:30])
        train_data_label.append(label_t[:30])
        test_data.append(data_t[30:])
        test_data_label.append(label_t[30:])

    train_data = np.concatenate(train_data)
    train_data_label = np.concatenate(train_data_label)
    test_data = np.concatenate(test_data)
    test_data_label = np.concatenate(test_data_label)

    x = tf.placeholder(tf.float32, [None, 4], name='input_x')
    label = tf.placeholder(tf.float32, [None, 3], name='input_y')

    net = slim.fully_connected(x, 4, activation_fn=tf.nn.relu)
    net = tf.contrib.layers.batch_norm(net)
    net = slim.fully_connected(net, 8, activation_fn=tf.nn.relu)
    net = tf.contrib.layers.batch_norm(net)
    net = slim.fully_connected(net, 8, activation_fn=tf.nn.relu)
    net = tf.contrib.layers.batch_norm(net)
    net = slim.fully_connected(net, 4, activation_fn=tf.nn.relu)
    net = tf.contrib.layers.batch_norm(net)
    y = slim.fully_connected(net, 3, activation_fn=tf.nn.softmax)

    loss = tf.reduce_sum(- label * tf.log(y), axis=1)
    loss = tf.reduce_mean(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    opt = tf.train.GradientDescentOptimizer(0.1)

    var_list_w = [var for var in tf.trainable_variables() if 'w' in var.name]
    var_list_b = [var for var in tf.trainable_variables() if 'b' in var.name]

    gradient_w = opt.compute_gradients(loss, var_list=var_list_w)
    gradient_b = opt.compute_gradients(loss, var_list=var_list_b)

    train_step = opt.apply_gradients(gradient_w + gradient_b)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for itr in range(600):
        sess.run(train_step, feed_dict={x: train_data, label: train_data_label})
        if itr % 30 == 0:
            acc1 = sess.run(accuracy, feed_dict={x: train_data, label: train_data_label})
            acc2 = sess.run(accuracy, feed_dict={x: test_data, label: test_data_label})
            print("step:{:6d}  train:{:.3f} test:{:.3f}".format(itr, acc1, acc2))

if __name__ == '__main__':
    data = pd.read_csv('./data/iris.data.csv')
    iris_tf2(data)




    

