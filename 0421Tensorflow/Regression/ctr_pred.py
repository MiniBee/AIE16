#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: ctr_pred.py
# @time: 2019/5/27 下午4:07
# @desc:

import pandas as pd


def sample_data():
    data = pd.read_csv('')
    sample = data.sample(20000)
    sample.to_csv('../data/ctr_sample.csv')


if __name__ == '__main__':
    sample_data()
    data = pd.read_csv('../data/ctr_sample.csv')



