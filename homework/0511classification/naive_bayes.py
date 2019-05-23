#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: naive_bayes.py
# @time: 2019/5/20 下午4:04
# @desc:

import numpy as np

class NaiveBayes():
    def N(self, x, mu, std):
        """
        标准正态分布
        """
        par = 1/(np.sqrt(2*np.pi)*std)
        return par*np.exp(-(x-mu)**2/2/std**2)
    def logN(self, x, class_type):
        """
        标准正态分布对数
        """
        if class_type==0:
            return np.log(self.N(x, self.mu1, self.std1))
        else:
            return np.log(self.N(x, self.mu2, self.std2))
    def fit(self, X, y):
        """
        训练过程为对于数据的统计
        """
        X1 = X[y==0]
        X2 = X[y==1]
        self.mu1 = np.mean(X1, axis=0)
        self.mu2 = np.mean(X2, axis=0)
        self.std1 = np.std(X1, axis=0)
        self.std2 = np.std(X2, axis=0)
    def predict_proba(self, xx):
        """
        预测过程
        """
        prb = []
        for x in xx:
            prb1_log = np.sum(self.logN(x, 0))
            prb2_log = np.sum(self.logN(x, 1))
            prb1 = np.exp(prb1_log)
            prb2 = np.exp(prb2_log)
            prb1 = prb1 / (prb1 + prb2)
            prb2 = prb2 / (prb1 + prb2)
            prb.append([prb1, prb2])
        return np.array(prb)

from sklearn.datasets import make_moons, make_circles, make_classification
#获取数据
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

method = NaiveBayes()
method.fit(X, y)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
Z = method.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()






