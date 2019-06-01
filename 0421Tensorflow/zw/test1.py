#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test1.py
# @time: 2019/6/1 上午9:15
# @desc:


import time
import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures


def strip_data(sentence, stopwords_file, cut_all=False):
    seg_list = jieba.cut(sentence, cut_all=cut_all)
    words = strip_word(seg_list, stopwords_file)
    return words


def strip_word(seg_list, stopwords_file):
    stopwords = [line for line in open(stopwords_file, 'r', encoding='utf-8').readlines()]
    stopwords.append('\t')
    words = []
    for word in seg_list:
        if word not in stopwords:
            if word != '\n':
                words.append(word)
    return words


def load_data(path='./data/6000/'):
    res_words = []
    file_names = []
    file_list = os.listdir(path)
    print(file_list)
    for file in file_list:
        file_name = file
        file = path + file
        with open(file, 'r', encoding='gbk', errors='ignore') as f:
            txt = f.read()
            words_ = strip_data(txt, './data/config/stopwords')
            res_words.append(' '.join(words_))
            file_names.append(file_name)
    return res_words, file_names


def countVectorizer_(words):
    countVectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1, max_features=20000)
    countVectorizer.fit(words)
    return countVectorizer

def tfidfVectorizer_(words):
    tfidfTransformer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1, max_features=20000)
    tfidfTransformer.fit(words)
    return tfidfTransformer


def lda(tokens_train):
    lda = LatentDirichletAllocation(n_components=100, max_iter=5, learning_method='online', learning_offset=50, random_state=0)
    lda.fit(tokens_train)
    return lda


if __name__ == '__main__':
    print('load data ...')
    st = time.perf_counter()
    jieba.load_userdict('./data/ud.dict')
    res_words, file_names = load_data()
    test_words, test_file = load_data('./data/10000/')
    ed = time.perf_counter()
    print('load data %s', ed - st)

    print('count vec ...')
    st = time.perf_counter()
    # countVectorizer = countVectorizer_(res_words)
    tfidfTransformer = tfidfVectorizer_(res_words)
    tokens_train = tfidfTransformer.transform(res_words)
    tokens_test = tfidfTransformer.transform(test_words)
    poly = PolynomialFeatures(degree=3)
    tokens_train = poly.fit_transform(tokens_train)
    tokens_test = poly.fit_transform(tokens_test)
    # tokens_train = countVectorizer.transform(res_words)
    # tokens_test = countVectorizer.transform(test_words)
    ed = time.perf_counter()
    print('count vec', ed - st)

    print('lda ...')
    st = time.perf_counter()
    lda = lda(tokens_train)
    # tokens_lda = lda.transform(tokens_train)
    # tokens_lda_test = lda.transform(tokens_test)`
    tokens_lda = tokens_train
    tokens_lda_test = tokens_test
    ed = time.perf_counter()
    print('lda ', ed - st)

    res_label = [1 if x.split('.')[0] == 'pos' else 0 for x in file_names]
    test_label = [1 if x.split('.')[0] == 'pos' else 0 for x in test_file]

    # x_train, x_test, y_train, y_test = train_test_split(tokens_lda, res_label, test_size=0.3, random_state=11)
    x_train, y_train = tokens_lda, res_label
    x_test, y_test = tokens_lda_test, test_label
    # print('gbdt ...')
    # st = time.perf_counter()
    # gbdt = GradientBoostingClassifier(learning_rate=0.5)
    # gbdt_param = {'max_depth': [5, 6, 7, 8], 'n_estimators': [100, 200, 300, 400]}
    # gbdt_grid = GridSearchCV(estimator=gbdt, param_grid=gbdt_param, cv=10, scoring='accuracy')
    # gbdt_grid.fit(x_train, y_train)
    # print(gbdt_grid.best_params_)
    # gbdt_ = gbdt_grid.best_estimator_
    # y_pred = gbdt_.predict(x_test)
    # print(gbdt_.score(x_train, y_train))
    # print(gbdt_.score(x_test, y_test))
    # ed = time.perf_counter()
    # print('gbdt ', ed - st)

    # print('random forest ...')
    # st = time.perf_counter()
    # rf = RandomForestClassifier(max_features='sqrt')
    # rf_param = {'n_estimators': [200, 300, 400]}
    # rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5, scoring='accuracy')
    # rf_grid.fit(x_train, y_train)
    # print(rf_grid.best_params_)
    # rf_ = rf_grid.best_estimator_
    # print(rf_.get_params())
    # print(rf_.score(x_train, y_train))
    # print(rf_.score(x_test, y_test))
    # ed = time.perf_counter()
    # print('random forest ', ed - st)

    print('xgboost ...')
    st = time.perf_counter()
    xgb = xgb.XGBClassifier(learning_rate=1)
    xgb_param = {'n_estimators': [600, 700, 800]}
    xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_param, cv=5, scoring='accuracy')
    xgb_grid.fit(x_train, y_train)
    print(xgb_grid.best_params_)
    xgb_ = xgb_grid.best_estimator_
    print(xgb_.get_params())
    print(xgb_.score(x_train, y_train))
    print(xgb_.score(x_test, y_test))
    ed = time.perf_counter()
    print('xgboost ', ed - st)

    # ExtraTreesClassifier
    print('et ...')
    st = time.perf_counter()
    et = ExtraTreesClassifier()
    et_param = {'n_estimators': [300, 400, 500, 600]}
    et_grid = GridSearchCV(estimator=et, param_grid=et_param, cv=5, scoring='accuracy')
    et_grid.fit(x_train, y_train)
    print(et_grid.best_params_)
    et_ = et_grid.best_estimator_
    print(et_.get_params())
    print(et_.score(x_train, y_train))
    print(et_.score(x_test, y_test))
    ed = time.perf_counter()
    print('et ', ed - st)

    # AdaBoostClassifier
    print('adb ...')
    st = time.perf_counter()
    adb = AdaBoostClassifier(learning_rate=0.1)
    adb_param = {'n_estimators': [50, 100, 150, 200]}
    adb_grid = GridSearchCV(estimator=adb, param_grid=adb_param, cv=5, scoring='accuracy')
    adb_grid.fit(x_train, y_train)
    print(adb_grid.best_params_)
    adb_ = adb_grid.best_estimator_
    print(adb_.get_params())
    print(adb_.score(x_train, y_train))
    print(adb_.score(x_test, y_test))
    ed = time.perf_counter()
    print('adb ', ed - st)

    # print('voting ...')
    # st = time.perf_counter()
    # vc = VotingClassifier(estimators=[('xgboost', xgb_), ('rf', rf_)], weights=[4, 6], voting = 'soft')
    # vc.fit(x_train, y_train)
    # y_pred = vc.predict(x_test)
    # print(vc.score(x_train, y_train))
    # print(vc.score(x_test, y_test))
    # ed = time.perf_counter()
    # print('voting ', ed - st)


