#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:16
LastEditTime: 2020-09-30 05:02:25
FilePath: /Assignment3-3_solution/generative/data.py
Desciption: Data processing for the generative module.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import sys
import os
import pandas as pd
import json

sys.path.append('..')
from config import root_path


with open(os.path.join(root_path, 'data/generative/train.tsv'), 'w') as train:
    count = 0
    with open(os.path.join(root_path, 'data/generative/LCCC-base_train.json'), 'r') as r:
        js_file = json.load(r)
        for item in js_file:
            if count >= 100000:
                break
            if len(item) == 2:
                line = '\t'.join(item)
                train.write(line)
                train.write('\n')
                count += 1
        

with open(os.path.join(root_path, 'data/generative/dev.tsv'), 'w') as dev:
    count = 0
    with open(os.path.join(root_path, 'data/generative/LCCC-base_valid.json'), 'r') as r:
        js_file = json.load(r)
        for item in js_file:
            
            if len(item) == 2:
                line = '\t'.join(item)
                dev.write(line)
                dev.write('\n')
                count += 1
                

with open(os.path.join(root_path, 'data/generative/test.tsv'), 'w') as test:
    count = 0
    with open(os.path.join(root_path, 'data/generative/LCCC-base_test.json'), 'r') as r:
        js_file = json.load(r)
        for item in js_file:
            
            if len(item) == 2:
                line = '\t'.join(item)
                test.write(line)
                test.write('\n')
                count += 1