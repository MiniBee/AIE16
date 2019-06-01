#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: nb_test.py
# @time: 2019/5/29 下午2:27
# @desc:

import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


if __name__ == '__main__':
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)

    # plt.scatter(data[:, 0], data[:, 1], c=y, s=200, marker='.', edgecolors='k')
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.title('train set')
    plt.show()

    plt.scatter(x_test[:, 0], x_test[:, 1])
    plt.title('test set')
    plt.show()

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
    plt.title('test set nb')
    plt.show()

    # NB 2
    x10 = np.random.randint(10, 20, 600)
    x10 = [x + np.random.random() for x in x10]
    x10.sort()
    x11 = [[x, 2 * x + 3 + np.random.random()] for x in x10]
    data1 = pd.DataFrame(x11, columns=['x0', 'x1'])
    # plt.scatter(data1.x0, data1.x1)
    data1['y'] = 0

    x20 = np.random.randint(9, 19, 400)
    x20 = [x + np.random.random() for x in x20]
    x20.sort()
    x21 = [[x, x + 7 + np.random.random()] for x in x20]
    data2 = pd.DataFrame(x21, columns=['x0', 'x1'])
    # plt.scatter(data2.x0, data2.x1)
    data2['y'] = 1
    # plt.show()

    x30 = np.random.randint(10, 20, 600)
    x30 = [x + np.random.random() for x in x30]
    x30.sort()
    x31 = [[x, 3 * x + 2 + np.random.random()] for x in x30]
    data3 = pd.DataFrame(x31, columns=['x0', 'x1'])
    # plt.scatter(data3.x0, data3.x1)
    data3['y'] = 2

    x40 = np.random.randint(9, 19, 400)
    x40 = [x + np.random.random() for x in x40]
    x40.sort()
    x41 = [[x, 4*x + 9 + np.random.random()] for x in x40]
    data4 = pd.DataFrame(x41, columns=['x0', 'x1'])
    # plt.scatter(data4.x0, data4.x1)
    data4['y'] = 3
    # plt.show()

    data = pd.concat([data1, data2, data3, data4], ignore_index=True)

    x_data = np.array(data[['x0', 'x1']])
    y_data = np.array(data['y'])

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.title('train set')
    plt.show()

    plt.scatter(x_test[:, 0], x_test[:, 1])
    plt.title('test set')
    plt.show()

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
    plt.title('test set nb')
    plt.show()

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
    plt.title('test set lr')
    plt.show()









