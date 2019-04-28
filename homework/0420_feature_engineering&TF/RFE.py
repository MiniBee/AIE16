#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: RFE.py
# @time: 2019/4/25 下午5:19
# @desc:

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


if __name__ == '__main__':
    iris = load_iris()
    print(iris['data'][:5])
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit(iris['data'], iris['target'])
    data = selector.transform(iris['data'])
    print(data[:5])
    print(selector.ranking_)




