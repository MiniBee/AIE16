#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: titanic_fe.py
# @time: 2019/4/26 下午4:35
# @desc:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb

xgb.XGBRegressor


pd.set_option('display.width', 1800)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 140)
pd.set_option('display.max_rows', 500)


def fill_age(data):

    all_df = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = all_df[all_df['Age'].notnull()].as_matrix()
    unknown_age = all_df[all_df['Age'].isnull()].as_matrix()

    x = known_age[:, 1:]
    y = known_age[:, 0]

    model = GradientBoostingRegressor()
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    y_pred = rfr.predict(unknown_age[:, 1::])
    data.loc[(data.Age.isnull()), 'Age'] = y_pred
    return data, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def fill_age_1(data, rfr):
    all_df = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = all_df[all_df['Age'].notnull()].values
    unknown_age = all_df[all_df['Age'].isnull()].values

    x = known_age[:, 1:]
    y = known_age[:, 0]

    y_pred = rfr.predict(unknown_age[:, 1::])
    data.loc[(data.Age.isnull()), 'Age'] = y_pred
    return data


if __name__ == '__main__':
    data_train = pd.read_csv('./data/train.csv', encoding='utf_8_sig')
    print(data_train.columns)
    print(data_train.info())

    AdaBoostClassifier()

    # plt.subplot2grid((2, 3), (1, 0), colspan=2)
    # data_train.Age.plot(kind='kde', grid=True, style='-k', title=u'age dis')
    #
    # plt.show()

    print(data_train.head())

    # 查看乘客获救情况 Survived=0/1
    # features1 = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    # fig = plt.figure(figsize=(10, 6))
    # for i, feature in enumerate(features1):
    #     s0 = data_train[feature][data_train['Survived'] == 0].value_counts()
    #     s1 = data_train[feature][data_train['Survived'] == 1].value_counts()
    #     df = pd.DataFrame({'Survived': s1, 'Unsurvived': s0})
    #     df.plot(kind='bar', grid=True, stacked=True, title=feature)
    #
    # plt.show()

    # data_orig = data_train.values
    # # age 处理
    data_train, rfr = fill_age(data_train)
    # data_train = set_Cabin_type(data_train)
    # data_train.to_csv("data/fix_data_tai1.csv")
    #
    # print(data_train.head())
    #

    # data_train = pd.read_csv("data/fix_data_tai1.csv")
    # dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    # dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    # dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    # dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    # df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # df.to_csv("data/fix_data_tai2.csv")
    # print(df.head())
    #
    df = pd.read_csv("data/fix_data_tai2.csv")
    df = df.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # train_df.to_csv("processed_titanic.csv" , encoding = "utf-8")
    # df.to_csv("data/fix_data_tai3.csv")
    print(df.head())

    x = df.values[:, 1:]
    y = df.values[:, 0]
    # model = LogisticRegression(C=100, penalty='l1', tol=1e-6)
    model = RandomForestClassifier()
    # # {'max_depth': 8, 'n_estimators': 500} 0.8271604938271605
    # m1_param_test = {
    #     'C': range(1, 10),
    #     'penalty': ['l1', 'l2']
    # }
    m2_param_test = {
        'max_depth': range(2, 10),
        'n_estimators': range(500, 1000, 100)
    }
    grid_search = GridSearchCV(estimator=model, param_grid=m2_param_test, scoring='accuracy', cv=10)
    # x = PolynomialFeatures(degree=2).fit_transform(x)
    # ldr = LinearDiscriminantAnalysis().fit(x, y)
    # x = ldr.transform(x)
    # x = StandardScaler().fit_transform(x)
    grid_search.fit(x, y)
    print(grid_search.best_params_, grid_search.best_score_)

    data_test = pd.read_csv('./data/test.csv', encoding='utf_8_sig')
    data_test = fill_age_1(data_test, rfr)
    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    df = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    df = df.filter(regex='Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    df.loc[(df.Fare.isnull()), 'Fare'] = df.Fare.mean()
    print(df.info())
    x = df.values
    print(df.head())
    # x = PolynomialFeatures(degree=2).fit_transform(x)
    # x = StandardScaler.fit_transform(x)
    # x = ldr.transform(x)
    y = grid_search.predict(x)
    data_test['Survived'] = y
    data_test[['PassengerId', 'Survived']].to_csv('gender_submission1.csv', index=0)


