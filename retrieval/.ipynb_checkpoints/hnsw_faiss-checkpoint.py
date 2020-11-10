#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-08-21 17:25:40
LastEditTime: 2020-08-27 22:12:52
FilePath: /Assignment3-1/retrieval/hnsw_faiss.py
Desciption: 使用Faiss训练hnsw模型。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import logging
import sys
import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import faiss

sys.path.append('..')
import config
from preprocessor import clean


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return: The sentence vector.
    '''
    arr = []
    for sent in clean(sentence):
        if (sent in w2v_model.wv.vocab.keys()):
            arr.append(w2v_model.wv.get_vector(sent))
        else:
            arr.append(np.random.randn(1,300))
    return np.mean(np.array(arr),axis=0).reshape(1,-1)


class HNSW(object):
    def __init__(self,
                 w2v_path,
                 ef=config.ef_construction,
                 M=config.M,
                 model_path=None,
                 data_path=None,
                 ):
        self.w2v_model = KeyedVectors.load(w2v_path)
        self.data = self.load_data(data_path)
        if model_path and os.path.exists(model_path):
            # 加载
            self.index = self.load_hnsw(model_path)
        elif data_path:
            # 训练
            self.index = self.build_hnsw(model_path,
                                         ef=ef,
                                         m=M)
        else:
            logging.error('No existing model and no building data provided.')

    def load_data(self, data_path):
        '''
        @description: 读取数据，并生成句向量
        @param {type}
        data_path：问答pair数据所在路径
        @return: 包含句向量的dataframe
        '''
        data = pd.read_csv(data_path)
        data['custom_vec']=data['custom'].apply(
            lambda x: wam(x, self.w2v_model)
        )
        data['custom_vec']=data['custom_vec'].apply(
            lambda x: float('nan') if x.shape[1]!=300 else x
        )
        data=data.dropna()
        return data

    def evaluate(self, vecs):
        '''
        @description: 评估模型。
        @param {type} vecs: The vectors to evaluate.
        @return {type} None
        '''

        logging.info('Evaluating.')
        num, dim = vecs.shape
        t = time.time()
        
        D, I = self.index.search(vecs.astype('float32'), 1)
        t1 = time.time()
        
        missing_rate = (I==-1).sum()/float(num)
        recall_at_1 = (I==np.arange(num)).sum()/float(num)
        
        print("\t %7.3f ms per query, Recall@1 %.4f, missing rate %.4f" % ((t1-t)*1000.0/num, recall_at_1, missing_rate))

    def build_hnsw(self, to_file, ef=2000, m=64):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        @return:
        '''
        logging.info('Building hnsw index.')
        dim = self.w2v_model.vector_size
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1,dim).astype('float32')
        
        #declare index model
        index = faiss.IndexHNSWFlat(dim,m)
        res = faiss.StandardGpuResources()
        faiss.index_cpu_to_gpu(res,0,index)
        index.hnsw.efConstruction = ef
        index.verbose = True
        
        index.add(vecs)
        faiss.write_index(index, to_file)
        return index

    def load_hnsw(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''
        logging.info('Loading hnsw index')
        hnsw = faiss.read_index(model_path)
        
        return hnsw

    def search(self, text, k):
        '''
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return: DataFrame contianing the customer input, assistance response
                and the distance to the query.
        '''
        test_vec = wam(clean(text), self.w2v_model)
        D , I = self.index.search(test_vec.astype('float32'),k)
        return pd.concat(
            (self.data.iloc[I[0]]['custom'].reset_index(),
            self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
            pd.DataFrame(D.reshape(-1,1), columns = ['q_distance'])),
            axis = 1)


if __name__ == "__main__":
    hnsw = HNSW(config.w2v_path,
                config.ef_construction,
                config.M,
                config.hnsw_path,
                config.train_path)
    test = '我要转人工'
    print(hnsw.search(test, k=4))
    eval_vecs = np.vstack(hnsw.data['custom_vec'].values).reshape(-1, 300)
    eval_vecs.astype('float32')
    hnsw.evaluate(eval_vecs[:10000])
