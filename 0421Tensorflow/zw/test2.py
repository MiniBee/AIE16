#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: test2.py
# @time: 2019/6/1 上午10:02
# @desc:

import jieba

if __name__ == '__main__':
    # jieba.load_userdict('./data/ud.dict')
    res = jieba.cut('超差，哈哈')
    print(' '.join(res))



