#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test1.py
# @time: 2019/4/13 上午11:42
# @desc:


import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing()


if __name__ == '__main__':
    # x = np.random.random(10000)
    # print(np.mean(x))
    x = sym.symbols('x')
    y = sym.exp(x) * x

    print(y.diff())

    # np.savez()
    # np.load()
    # np.savez_compressed()
    # np.random.random()
    # np.random.normal()

    # plt.matshow()  绘制矩阵

    # np.linalg.svd()  SVD


