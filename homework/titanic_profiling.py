#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: titanic_profiling.py
# @time: 2019/4/29 上午9:58
# @desc:


import pandas as pd
import pandas_profiling as pdf


if __name__ == '__main__':
    data_train = pd.read_csv('./data/train.csv', encoding='utf_8_sig')

