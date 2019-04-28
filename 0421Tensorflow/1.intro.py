#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 1.intro.py
# @time: 2019/4/21 上午9:42
# @desc:


import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    # 简单计算
    # a1 = tf.constant(np.ones([4, 4]))
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # res = sess.run(a1)
    # print(res)

    # 展示相乘结果
    # a1 = tf.get_variable('a1', [4, 4])
    # a2 = tf.get_variable('a2', [4, 4])
    # c1 = tf.matmul(a1, a2)
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # res = sess.run(c1)

    # 定义、展示计算图
    # a1 = tf.Variable()...
    # graph = tf.Graph()  # 定义计算图,否则默认，Session(graph=默认值)
    # with graph.as_default():
    #     a1 = tf.get_variable('a1', [4, 4])
    #     a2 = tf.get_variable('a2', [4, 4])
    #     c1 = tf.matmul(a1, a2)
    #     init = tf.global_variables_initializer()
    #
    # tf.summary.FileWriter('my_first_demo', graph=graph)  # 在浏览器上展示计算图
    # # tensorboard --logdir=my_first_demo  # 展示demo命令
    # sess = tf.Session(graph=graph)
    # sess.run(init)
    # res = sess.run(c1)
    # sess.close()
    # print(res)


    # 变量作用域
    # graph = tf.Graph()  # 定义计算图,否则默认，Session(graph=默认值)
    # with graph.as_default():
    #     # 限定a1，a2的作用域，即在前边加上"fole1/"
    #     with tf.variable_scope('fold1'):
    #         a1 = tf.get_variable('a1', [4, 4])
    #         a2 = tf.get_variable('a2', [4, 4])
    #     print(a1.name, a2.name)
    #     c1 = tf.matmul(a1, a2)
    #     init = tf.global_variables_initializer()
    #
    # tf.summary.FileWriter('my_first_demo', graph=graph)  # 在浏览器上展示计算图
    # # tensorboard --logdir=my_first_demo  # 展示demo命令
    # sess = tf.Session(graph=graph)
    # sess.run(init)
    # res = sess.run(c1)
    # sess.close()
    # print(res)


    # 定义placeholder, 用于从外界接受样本
    # graph = tf.Graph()  # 定义计算图,否则默认，Session(graph=默认值)
    # with graph.as_default():
    #     # 限定a1，a2的作用域，即在前边加上"fole1/"
    #     with tf.variable_scope('fold1'):
    #         a1 = tf.get_variable('a1', [4, 4])
    #         a2 = tf.get_variable('a2', [4, 4])
    #     print(a1.name, a2.name)
    #     c1 = tf.matmul(a1, a2)
    #     init = tf.global_variables_initializer()
    #
    # tf.summary.FileWriter('my_first_demo', graph=graph)  # 在浏览器上展示计算图
    # # tensorboard --logdir=my_first_demo  # 展示demo命令
    # sess = tf.Session(graph=graph)
    # sess.run(init)
    # res = sess.run(c1)
    # sess.close()
    # print(res)

    # 练习
    batch_size = 32
    learning_rate = 0.05
    inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])
    target = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])

    w = tf.get_variable('w', shape=[1, 1])
    b = tf.get_variable('w', shape=[1])

    y = tf.matmul(inputs, w) + b

    # loss函数
    loss = tf.reduce_mean((y - target) ** 2)
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # 梯度下降优化去
    # opt.compute_gradients(loss=loss)  # 计算梯度
    step = opt.minimize(loss=loss)

    # 赋值

    file = np.load('')
    x = file['X']
    d = file['d']

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(500):
        inx = np.random.randint(1, 1000, batch_size)
        inx = x[inx]
        ind = d[ind]
        sess.run(step, feed_dict={inputs: inx, target: ind})
    sess.close()






