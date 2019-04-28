#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: iris_fe.py
# @time: 2019/4/25 下午5:34
# @desc:

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def train_model(x, y, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    return model, score


if __name__ == '__main__':
    iris = load_iris()
    x = iris['data']
    y = iris['target']

    k = 5
    # base_line, 0.9666666666666668
    model, score = train_model(x, y)
    print('base line')
    print(score.mean())

    # data view
    plt.figure(figsize=(10, 10))
    for i in range(len(x[0])):
        for j in range(len(x[0])):
            xlabel = x[:, i]
            ylabel = x[:, j]
            plt.subplot2grid((4, 4), (i, j))
            plt.scatter(xlabel, ylabel, edgecolors='k')
    plt.show()

    # 数据分布情况， 均值，3倍标准差
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(x[0])):
        xlabel = x[:, i]
        a = pd.Series(np.array(xlabel))
        ax = fig.add_subplot(4, 1, 1+i)
        std = a.std()
        u = a.mean()
        a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
        plt.axvline(u, color='r', linestyle="--", alpha=0.8)
        plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
        plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)

    plt.show()

    # 特征工程 StandardScaler, 0.9533333333333334
    x_standard = StandardScaler().fit_transform(x)
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(x[0])):
        xlabel = x_standard[:, i]
        a = pd.Series(np.array(xlabel))
        ax = fig.add_subplot(4, 1, 1+i)
        std = a.std()
        u = a.mean()
        a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
        plt.axvline(u, color='r', linestyle="--", alpha=0.8)
        plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
        plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)

    plt.show()
    model, score = train_model(x_standard, y)
    print('StandardScaler')
    print(score.mean())

    # 特征工程 MinmaxScaler, 0.9533333333333334
    x_standard = MinMaxScaler().fit_transform(x)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1+i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()

    model, score = train_model(x_standard, y)
    print('MinmaxScaler')
    print(score.mean())

    # 特征工程 Normalizer, 0.96
    x_standard = Normalizer().fit_transform(x)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()

    model, score = train_model(x_standard, y)
    print(u'Normalizer')
    print(score.mean())

    # 特征工程 PCA,
    x_standard = PCA(n_components=2).fit_transform(x)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()

    model, score = train_model(x_standard, y)
    print('PCA')
    print(score.mean())

    # Normalizer + PCA 0.9733333333333334
    x_standard = Normalizer().fit_transform(x)
    x_standard = PCA(n_components=2).fit_transform(x_standard)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()
    model, score = train_model(x_standard, y)
    print('Normalizer + PCA')
    print(score.mean())

    # StandardScaler + PCA 0.9066666666666666
    x_standard = StandardScaler().fit_transform(x)
    x_standard = PCA(n_components=2).fit_transform(x_standard)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()
    model, score = train_model(x_standard, y)
    print('StandardScaler + PCA')
    print(score.mean())

    # MinMaxScaler + PCA 0.9733333333333334
    x_standard = MinMaxScaler().fit_transform(x)
    x_standard = PCA(n_components=2).fit_transform(x_standard)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()
    model, score = train_model(x_standard, y)
    print('MinMaxScaler + PCA')
    print(score.mean())

    # Normalizer + PCA + PolynomialFeatures 0.9733333333333334
    x_standard = PolynomialFeatures(degree=2).fit_transform(x)
    x_standard = Normalizer().fit_transform(x_standard)
    # x_standard = PCA(n_components=2).fit_transform(x_standard)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()
    model, score = train_model(x_standard, y)
    print('Normalizer + PolynomialFeatures(degree=2) + PCA')
    print(score.mean())

    # Normalizer + RFE + PolynomialFeatures 0.9266666666666667
    # x_standard = PolynomialFeatures(degree=2).fit_transform(x_standard)
    x_standard = Normalizer().fit_transform(x)
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit(x_standard, y)
    x_standard = selector.transform(x_standard)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()
    model, score = train_model(x_standard, y)
    print('Normalizer + RFE')
    print(score.mean())

    # Normalizer + LDA + PolynomialFeatures 0.9266666666666667
    # x_standard = PolynomialFeatures(degree=2).fit_transform(x_standard)
    x_standard = MinMaxScaler().fit_transform(x)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x_standard, y)
    x_standard = lda.transform(x_standard)
    # fig = plt.figure(figsize=(10, 6))
    # for i in range(len(x_standard[0])):
    #     xlabel = x_standard[:, i]
    #     a = pd.Series(np.array(xlabel))
    #     ax = fig.add_subplot(4, 1, 1 + i)
    #     std = a.std()
    #     u = a.mean()
    #     a.plot(kind='kde', grid=True, style='-k', title=u'mi du qu xian')
    #     plt.axvline(u, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u + 3 * std, color='r', linestyle="--", alpha=0.8)
    #     plt.axvline(u - 3 * std, color='r', linestyle="--", alpha=0.8)
    #
    # plt.show()
    model, score = train_model(x_standard, y)
    print('Normalizer + LDA')
    print(score.mean())


