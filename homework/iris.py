#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: iris.py.py
# @time: 2019/4/16 下午9:57
# @desc:

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import pandas_profiling as pdp
import pandas as pd


def load_data():
    return datasets.load_iris()


if __name__ == '__main__':
    iris_data = load_data()
    x, y = iris_data.data, iris_data.target
    # a = pd.DataFrame(x)
    # pdp.ProfileReport(a).to_file('./iris.html')
    print([x for x in zip(x, y)])
    print(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    model = Pipeline(steps=[
        ('poly', PolynomialFeatures()),
        ('classifier', KNeighborsClassifier())
    ])

    for d in range(1, 4):
        k_score = []
        for k in range(1, 31):
            model.set_params(poly__degree=d, classifier__n_neighbors=k)
            model.fit(x_train, y_train)
            y_hat = model.predict(x_test)
            score = model.score(x_test, y_test)
            k_score.append(score)
            print(d, k)
            print(score)
        plt.plot(range(1, 31), k_score)
        plt.show()

    k_range = range(1, 31)
    k_score = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, x, y, cv=10, scoring='accuracy')
        k_score.append(score.mean())

    plt.plot(k_range, k_score)
    plt.show()



