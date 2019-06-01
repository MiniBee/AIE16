#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: happiness_index_v2.py
# @time: 2019/5/24 上午11:18
# @desc: 0.468092809384329 -> 0.4465355019845877 -> 0.44253712930299904

import pandas as pd
import numpy as np

import pandas_profiling as pdp
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error


from scipy import stats

pd.set_option('display.width', 1800)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 500)


# -1, -2的情况待考虑
def patient_age_category(data):
    data['survey_time'] = data['survey_time'].apply(lambda x: int(x))
    data['f_age'] = data['survey_time'] - data['f_birth']
    data['m_age'] = data['survey_time'] - data['m_birth']
    return data


def age_category(data):
    data['survey_time'] = data['survey_time'].apply(lambda x: int(x))
    data['Age'] = data['survey_time'] - data['birth']
    data.loc[data['Age'] <= 18, 'Age'] = 0
    data.loc[(data['Age'] > 18) & (data['Age'] <= 36), 'Age'] = 1
    data.loc[(data['Age'] > 36) & (data['Age'] <= 18*3), 'Age'] = 2
    data.loc[(data['Age'] > 18*3) & (data['Age'] <= 18*4), 'Age'] = 3
    data.loc[(data['Age'] > 18*4) & (data['Age'] <= 18*5), 'Age'] = 4
    data.loc[data['Age'] > 18*5, 'Age'] = 5
    return data


def city_category(data):
    city_c = pd.get_dummies(data['city'], prefix='city')
    data = pd.concat([data, city_c], axis=1)
    return data


def fill_edu_status(data):
    data['edu_status'] = data[['edu_status']].fillna(data[['city', 'Age', 'edu_status']]
                                                     .groupby(['city', 'Age'])
                                                     .agg(lambda x: stats.mode(x)[0][0]).reset_index())
    return data


def ch_count(data):
    son_c = data['son'] if data['son'] != -8 else 0
    dau_c = data['daughter'] if data['daughter'] != -8 else 0
    if data['son'] == -8 and data['daughter'] == -8:
        return -8
    else:
        return son_c + dau_c


def income_category(data):
    if data['family_income'] > 250000:
        return 6
    elif data['family_income'] > 200000:
        return 5
    elif data['family_income'] > 150000:
        return 4
    elif data['family_income'] > 100000:
        return 3
    elif data['family_income'] > 50000:
        return 2
    else:
        return 1

def income_category2(data):
    if data['income'] > 80000:
        return 5
    elif data['income'] > 60000:
        return 4
    elif data['income'] > 40000:
        return 3
    elif data['income'] > 20000:
        return 2
    else:
        return 1


def feature_e(data):
    data['survey_time'] = data['survey_time'].apply(lambda x: int(x[:4]))
    data['Age'] = data['survey_time'] - data['birth']
    data['ch'] = data.apply(lambda x: ch_count(x), axis=1)
    data['family_income'] = data[['family_income']].fillna(data[['city', 'Age', 'edu_status', 'family_income']].groupby(['city', 'Age', 'edu_status']).transform('mean'))
    data['family_income_category'] = data.apply(lambda x: income_category(x), axis=1)
    data['income_category'] = data.apply(lambda x: income_category2(x), axis=1)
    data = fill_edu_status(data)
    data = city_category(data)
    data = age_category(data)
    return data


if __name__ == '__main__':
    data = pd.read_csv('./data/happiness_train_complete.csv', encoding='latin-1')
    test_data = pd.read_csv('./data/happiness_test_complete.csv', encoding='latin-1')

    data = data.loc[data['happiness'] != -8]
    data = feature_e(data)
    print(data.count())

    test_data = feature_e(test_data)

    features = data.dropna(axis=1, how='any').columns
    # print(features)
    features = features.values.tolist()
    features.remove('happiness')
    # # features.extend(['Age', 'work_exper'])
    # remove_list = ['id', 'happiness', 'survey_time', 'birth', 'city', 'county', 'gender', 'province',
    #                'invest_0', 'invest_1', 'invest_2', 'invest_3', 'invest_4', 'invest_5', 'invest_6', 'invest_7', 'invest_8',
    #                'property_5', 'property_6', 'property_0'
    #                ]
    # for a in remove_list:
    #     features.remove(a)
    features = ['equity', 'depression', 'class', 'family_status', 'floor_area', 'family_income', 'weight_jin', 'status_peer', 'height_cm', 'public_service_7', 'birth', 'health', 'inc_exp', 'income', 'class_10_after', 'public_service_6', 'relax', 'public_service_5', 'family_m', 'public_service_2', 'public_service_3', 'public_service_1', 'public_service_9', 'public_service_8', 'health_problem', 'class_14', 'f_birth', 'class_10_before', 'public_service_4', 'media_4', 'view', 'neighbor_familiarity', 'm_birth', 'inc_ability', 'leisure_9', 'edu', 'trust_7', 'trust_1', 'status_3_before', 'trust_8', 'trust_10', 'leisure_1', 'ch', 'leisure_3', 'daughter', 'socialize', 'house', 'trust_9', 'trust_2', 'f_edu', 'trust_12', 'trust_6', 'leisure_7', 'media_3', 'trust_5', 'trust_13', 'work_exper', 'f_work_14', 'm_work_14', 'socia_outing', 'trust_3', 'leisure_8', 'marital', 'leisure_6', 'trust_11', 'trust_4', 'm_edu', 'leisure_11', 'son', 'learn', 'media_6', 'religion_freq', 'leisure_4', 'leisure_2', 'leisure_5', 'media_2', 'media_1', 'insur_2', 'hukou', 'leisure_10', 'insur_3', 'media_5', 'city_12', 'insur_1', 'religion', 'f_political', 'property_2', 'income_category', 'insur_4', 'Age', 'leisure_12', 'political', 'nationality', 'gender', 'property_1', 'city_64', 'car', 'city_69', 'city_40', 'survey_type', 'city_46', 'family_income_category', 'city_82', 'city_16', 'property_3', 'city_23', 'property_8', 'property_4', 'city_61', 'city_45', 'city_65', 'city_68', 'city_11', 'city_22', 'city_42', 'city_7', 'city_54', 'city_59', 'city_41', 'city_81', 'm_political', 'city_84', 'invest_2', 'city_39', 'city_8', 'city_15', 'city_85', 'city_48', 'city_37', 'city_29', 'city_4', 'city_80', 'city_63', 'city_27', 'city_87', 'city_19', 'city_52', 'city_35', 'city_57', 'city_14', 'property_7', 'city_62', 'city_49', 'invest_1', 'city_24', 'city_53', 'city_36', 'invest_0', 'city_51', 'city_86', 'city_25', 'city_30', 'city_20', 'city_67', 'city_18', 'city_10', 'city_13']
    print(data[features].head())
    x_data = data[features]
    y_data = data['happiness']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    x_sub = test_data[features]
    # random forest
    print('-----random forest------')
    rf_est = RandomForestRegressor(n_estimators=300)
    # folds = KFold(n_splits=10, shuffle=True)
    # for fold_, (trn_inx, val_inx) in enumerate(folds.split(x_data, y_data)):
    #     pass
    rf_est.fit(x_train, y_train)
    y_pred1 = rf_est.predict(x_test)
    y_sub1 = rf_est.predict(x_sub)
    a = [i for i in zip(features, rf_est.feature_importances_)]
    a.sort(key=lambda x: x[1], reverse=True)
    print(a)
    features = [i[0] for i in a]
    print(features)
    print(mean_squared_error(y_test, y_pred1))

    # gbdt
    gbdt_est = GradientBoostingRegressor(n_estimators=300)
    gbdt_param = {
        'max_depth': range(2, 5),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
    }
    gbdt_gsr = GridSearchCV(estimator=gbdt_est, param_grid=gbdt_param, scoring='neg_mean_squared_error', cv=10,
                            n_jobs=50)

    gbdt_gsr.fit(x_train, y_train)
    y_pred2 = gbdt_gsr.predict(x_test)
    y_sub2 = gbdt_gsr.predict(x_sub)
    y_pred = (np.array(y_pred2) + np.array(y_pred1)) / 2
    print(mean_squared_error(y_test, y_pred2))

    # xgboost
    xgb_est = xgb.XGBRegressor(n_estimators=300, n_jobs=50)
    xgb_param = {
        'max_depth': range(2, 5),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
    }
    xgb_gsr = GridSearchCV(estimator=xgb_est, param_grid=xgb_param, scoring='neg_mean_squared_error', cv=10, n_jobs=50)
    xgb_gsr.fit(x_train, y_train)
    y_pred3 = xgb_gsr.predict(x_test)
    y_sub3 = xgb_gsr.predict(x_sub)
    print(mean_squared_error(y_test, y_pred3))

    y_pred = (np.array(y_pred) + np.array(y_pred3)) / 2
    y_sub = (np.array(y_sub1) + np.array(y_sub2) + np.array(y_sub3)) / 3

    print(mean_squared_error(y_test, y_pred))

    test_data['happiness'] = y_sub
    test_data[['id', 'happiness']].to_csv('./data/happiness_submit.csv', index=False)



