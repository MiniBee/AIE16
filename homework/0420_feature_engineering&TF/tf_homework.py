#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: tf_homework.py
# @time: 2019/4/28 上午10:18
# @desc:


import numpy as np


if __name__ == '__main__':
    data = np.load('./data/homework.npz')
    x = data['X']
    y = data['d']
    print(x, y)


