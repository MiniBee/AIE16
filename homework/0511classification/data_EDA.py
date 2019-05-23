#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: data_EDA.py
# @time: 2019/5/17 上午10:49
# @desc:

from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.show()




