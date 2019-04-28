#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: plt_plot.py
# @time: 2019/4/25 下午5:25
# @desc:

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    x = [x for x in range(-10, 11)]
    y = [i ** 3 + 10 for i in x]

    plt.figure(22)
    plt.subplot(211)
    plt.plot(x, y, lw=3)
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(212)
    X = np.linspace(-2, 2, 20)
    Y = 2 * X + 1
    plt.scatter(X, Y)
    plt.show()


