#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: diabetes_regression.py
# @time: 2019/5/29 上午10:10
# @desc:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def load_data():
    diabetes = datasets.load_diabetes()
    return diabetes.data, diabetes.target


def model_train(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


if __name__ == '__main__':
    x_data, y_data = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    lrm = model_train(LinearRegression(), x_train, y_train)
    y_pred = lrm.predict(x_test)
    print('Mean squared error: %s' % mean_squared_error(y_test, y_pred))
    print('R2 score : %s' % r2_score(y_test, y_pred))

    rfm = model_train(RandomForestRegressor(n_estimators=300), x_train, y_train)
    y_pred = rfm.predict(x_test)
    print('Mean squared error: %s' % mean_squared_error(y_test, y_pred))
    print('R2 score : %s' % r2_score(y_test, y_pred))

    gbd = model_train(GradientBoostingRegressor(max_depth=1, n_estimators=100), x_train, y_train)
    y_pred = gbd.predict(x_test)
    print('Mean squared error: %s' % mean_squared_error(y_test, y_pred))
    print('R2 score : %s' % r2_score(y_test, y_pred))

