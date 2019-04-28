#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test.py
# @time: 2019/4/28 下午2:22
# @desc:


import tensorflow as tf
batch_size = None
learn_rate = 1e-3
#输入inputs:X
#标签target:d
inputs = tf.placeholder(tf.float32, [batch_size, 2])
target = tf.placeholder(tf.float32, [batch_size, 1])
# 建立的模型
h1 = tf.layers.dense(inputs, 6, activation=tf.nn.relu)


# loss函数
loss = (y-target) ** 2
loss = tf.reduce_mean(loss)

opt = tf.train.GradientDescentOptimizer(learn_rate)
step = opt.minimize(loss)

#a3 = tf.get_variable("a1", [4, 4])
#tf.summary.FileWriter("myfirstdemo", graph=graph)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

import numpy as np
file = np.load("homework.npz")
X = file['X']
d = file['d']
for itr in range(500):
    idx = np.random.randint(0, 1000, 32)
    inx = X[idx]
    ind = d[idx]
    st, ls = sess.run([step, loss], feed_dict={inputs:inx, target:ind})
    print(itr, ls)

import matplotlib.pyplot as plt
w, w2, b = sess.run([w, w2, b])
x = np.linspace(-2, 4, 100)
y = w[0, 0] * x + b[0] + w2[0, 0] * x**2
plt.scatter(X[:, 0], d[:, 0])
plt.plot(x, y, lw=3, color="#000000")
plt.show()


