#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: GD.py
# @time: 2019/5/27 下午3:16
# @desc:

import matplotlib.pyplot as plt
import random


error_array = []
epoch_array = []


def BGD(x, y):
    diff = 0
    theta0 = 0
    theta1 = 0
    theta2 = 0

    alpha = 0.01

    sum0 = 0
    sum1 = 0
    sum2 = 0
    m = len(x)

    epoch = 0

    while True:
        for i in range(m):
            diff = y[i] - (theta0 * 1 + theta1 * x[i][0] + theta2 * x[i][1])
            sum0 -= -alpha * diff * 1
            sum1 -= -alpha * diff * x[i][0]
            sum2 -= -alpha * diff * x[i][1]

        theta0 += sum0 / m
        theta1 += sum1 / m
        theta2 += sum2 / m

        sum0 = 0
        sum1 = 0
        sum2 = 0

        error1 = 0
        for i in range(m):
            error1 += (y[i] - (theta0 + theta1 * x[i][0] + theta2 * x[i][1])) ** 2
        error1 = error1 / m

        epoch_array.append(epoch)
        error_array.append(error1)

        if epoch == 2000:
            break
        else:
            error0 = error1
        epoch += 1
        print(' theta0 : %s, theta1: %s, theta2: %s, bdg error1: %s, epoch: %s ' % (theta0, theta1, theta2, error1, epoch))


def SGD(x, y):
    diff = 0
    theta0 = 0
    theta1 = 0
    theta2 = 0

    alpha = 0.001
    epoch = 0
    error1 = 0
    m = len(x)
    while True:
        i = random.randint(0, m - 1)
        diff = y[i] - theta0 * 1 - theta1 * x[i][0] - theta2 * x[i][1]

        theta0 += alpha * diff
        theta1 += alpha * diff * x[i][0]
        theta2 += alpha * diff * x[i][1]

        for i in range(m):
            error1 += (y[i] - (theta0 + theta1 * x[i][0] + theta2 * x[i][1])) ** 2
        error1 = error1 / m
        error_array.append(error1)
        epoch_array.append(epoch)
        if epoch == 200:
            break
        else:
            pass
        epoch += 1
        print(' theta0 : %s, theta1: %s, theta2: %s, bdg error1: %s, epoch: %s ' % (theta0, theta1, theta2, error1, epoch))


def mini_BGD(x, y):
    batchsize = 3
    error1 = 0
    epoch = 0
    alpha = 0.01
    theta0 = 0
    theta1 = 0
    theta2 = 0
    diff = 0
    m = len(x)
    sum0 = 0
    sum1 = 0
    sum2 = 0
    while True:
        for j in range(batchsize):
            i = random.randint(0, m-1)
            diff = y[i] - theta0 - theta1 * x[i][0] - theta2 * x[i][1]
            sum0 += alpha * diff
            sum1 += alpha * diff * x[i][0]
            sum2 += alpha * diff * x[i][1]
        theta0 += sum0 / batchsize
        theta1 += sum1 / batchsize
        theta2 += sum2 / batchsize
        sum0 = 0
        sum1 = 0
        sum2 = 0

        for i in range(m):
            error1 += (y[i] - (theta0 + theta1 * x[i][0] + theta2 * x[i][1])) ** 2
        error1 = error1 / m

        epoch_array.append(epoch)
        error_array.append(error1)

        if epoch == 2000:
            break
        epoch += 1
        print(' theta0 : %s, theta1: %s, theta2: %s, bdg error1: %s, epoch: %s ' % (
        theta0, theta1, theta2, error1, epoch))


if __name__ == '__main__':
    x = [(0., 3), (1., 3), (2., 3), (3., 2), (4., 4), (0., 3), (1., 3.1), (2., 3.5), (3., 2.1), (4., 4.2)]
    y = [95.364, 97.217205, 75.195834, 60.105519, 49.342380, 100.364, 100.217205, 100.195834, 100.105519, 12.342380]
    # BGD(x, y)
    # SGD(x, y)
    # mini_BGD(x, y)
    plt.plot(epoch_array, error_array)
    plt.show()


