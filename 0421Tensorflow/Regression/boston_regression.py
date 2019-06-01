#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: boston_regression.py
# @time: 2019/5/29 上午10:38
# @desc:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error


def load_data():
    data = pd.read_csv('../data/regression/housing.csv')
    return data


def data_eda(data):
    rm = data['RM']
    medv = data['MEDV']
    plt.scatter(rm, medv, c='b')
    plt.show()
    lstat = data['LSTAT']
    plt.scatter(lstat, medv, c='c')
    plt.show()
    ptratio = data['PTRATIO']
    plt.scatter(ptratio, medv, c='g')
    plt.show()


def model_gridsearch(estimator, param_grid, scoring, x_train, y_train, cv=10):
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv)
    grid.fit(x_train, y_train)
    print('best param' + str(grid.best_params_))
    print('best score' + str(grid.best_score_))
    return grid.best_estimator_


if __name__ == '__main__':
    data = load_data()
    print(data.head())
    # data_eda(data)
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.3)
    model = model_gridsearch(GradientBoostingRegressor(), param_grid={'max_depth': [1,2,3,4]},
                             scoring=make_scorer(r2_score), x_train=x_train, y_train=y_train)
    y_pred = model.predict(x_test)
    print("Optimal model has R^2 score {:,.2f} on test data".format(r2_score(y_test, y_pred)))
    print(mean_squared_error(y_test, y_pred))


