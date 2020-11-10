#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-08-10 15:53:39
LastEditTime: 2020-08-27 21:30:51
FilePath: /Assignment3-1/retrieval/hnsw_hnswlib.py
Desciption: 使用hnswlib训练hnsw模型。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import logging
import sys
import os

sys.path.append('..')

import hnswlib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from retrieval.preprocessor import clean


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return: Sentence embeded vector.
    '''

    return


class HNSW(object):
    def __init__(self,
                 w2v_path,
                 data_path=None,
                 ef=ef_construction,
                 M=M,
                 model_path=None):
        self.w2v_model = KeyedVectors.load(w2v_path)

        self.data = self.data_load(data_path)
        if model_path is not None:
            # 加载
            self.hnsw = self.load_hnsw(model_path)
        else:
            # 训练
            self.hnsw = \
                self.build_hnsw(os.path.join(root_path, 'model/hnsw.bin'),
                                ef=ef,
                                m=M)

    def data_load(self, data_path):
        '''
        @description: 读取数据，并生成句向量
        @param {type}
        data_path：问答pair数据所在路径
        @return: 包含句向量的dataframe
        '''

        return data

    def build_hnsw(self, to_file, ef=2000, m=64):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        @return:
        '''

        return index

    def load_hnsw(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''

        return index

    def search(self, text, k=5):
        '''
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return:
        '''

        return


if __name__ == "__main__":
    hnsw = HNSW(config.w2v_path),
                config.train_path, config.ef_construction, config.M)
    test = '我要转人工'
    print(hnsw.search(test, k=10))
