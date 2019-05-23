#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: happiness_index_v1.py
# @time: 2019/5/23 上午10:23
# @desc:

import pandas as pd

import pandas_profiling as pdp
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error

pd.set_option('display.width', 1800)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 140)
pd.set_option('display.max_rows', 500)


def age_category(data):
    data['survey_time'] = data['survey_time'].apply(lambda x: int(x[:4]))
    data['Age'] = data['survey_time'] - data['birth']
    data.loc[data['Age'] <= 18, 'Age'] = 0
    data.loc[(data['Age'] > 18) & (data['Age'] <= 36), 'Age'] = 1
    data.loc[(data['Age'] > 36) & (data['Age'] <= 18*3), 'Age'] = 2
    data.loc[(data['Age'] > 18*3) & (data['Age'] <= 18*4), 'Age'] = 3
    data.loc[(data['Age'] > 18*4) & (data['Age'] <= 18*5), 'Age'] = 4
    data.loc[data['Age'] > 18*5, 'Age'] = 5
    return data


if __name__ == '__main__':
    data = pd.read_csv('./data/happiness_train_complete.csv', encoding='latin-1')
    data = age_category(data)

    features = (data.corr()['happiness'][abs(data.corr()['happiness']) > 0.05]).index
    features = features.values.tolist()
    features.extend(['Age', 'work_exper'])

    remove_list = ['happiness', 'join_party', 'edu_yr', 'social_friend', 'work_type', 'marital_1st', 'marital_now',
                   's_edu', 's_political', 's_hukou']
    for a in remove_list:
        features.remove(a)

    x_data = data[features]
    y_data = data['happiness']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7)
    print(x_data.info())

    # random forest
    rf_est = RandomForestRegressor(n_estimators=100)

    # gbdt
    gbdt_est = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100)
    gbdt_param = {
        'max_depth': range(2, 5)
    }
    gbdt_gsr = GridSearchCV(estimator=gbdt_est, param_grid=gbdt_param, scoring='neg_mean_squared_error', cv=10, n_jobs=50)

    # xgboost
    xgb_est = xgb.XGBRegressor(n_estimators=100, n_jobs=50)
    xgb_param = {
        'max_depth': range(2, 5),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
    }
    xgb_gsr = GridSearchCV(estimator=xgb_est, param_grid=xgb_param, scoring='neg_mean_squared_error', cv=10, n_jobs=50)

    # voting
    voting_est = VotingClassifier(estimators=[('xgb', xgb_gsr), ('gbdt', gbdt_gsr), ('rf', rf_est)],
                                  voting='soft',
                                  n_jobs=50)
    voting_param = {
        'weights': [[3, 4, 3], [2, 3, 5], [5, 3, 2]]
    }
    voting_gsr = GridSearchCV(estimator=voting_est, param_grid=voting_param, n_jobs=50)

    voting_gsr.fit(x_train, y_train)

    print(voting_gsr.score(x_train, y_train))
    print(voting_gsr.score(x_test, y_test))

    y_pred = voting_gsr.predict(x_test)
    print(mean_squared_error(y_test, y_pred))


