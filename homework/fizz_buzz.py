#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: fizz_buzz.py
# @time: 2019/4/15 下午9:38
# @desc:


from sklearn.linear_model import LogisticRegression
import numpy as np


def check_fizz_buzz(x):
    if x % 3 == 0 and x % 5 == 0:
        return 'fizzbuzz'
    if x % 3 == 0:
        return 'fizz'
    if x % 5 == 0:
        return 'buzz'
    else:
        return x


def get_label(x):
    if x % 3 == 0 and x % 5 == 0:
        return 0
    if x % 3 == 0:
        return 1
    if x % 5 == 0:
        return 2
    else:
        return 3


if __name__ == '__main__':
    print('='*15, 'Rule Based', '='*15)
    data_list = [x for x in range(1, 101)]
    res = [x for x in map(check_fizz_buzz, data_list)]
    print(data_list)
    print(res)

    print('=' * 15, 'ML', '=' * 15)
    x_train = [x for x in map(lambda x: [x % 3, x % 5, x % 15], [i for i in range(1, 101)])]
    y_train = [x for x in map(get_label, [i for i in range(1, 101)])]
    print(x_train)
    print(y_train)

    x_test = [x for x in map(lambda x: [x % 3, x % 5, x % 15], [i for i in range(102, 200)])]
    y_test = [x for x in map(get_label, [i for i in range(102, 200)])]

    for i in np.arange(0.1, 1.1, 0.1):
        print('C = %s' % i)
        model = LogisticRegression(C=i)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(model.score(x_test, y_test))
        print(y_test)
        print(y_pred)

