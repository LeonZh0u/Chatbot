#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-08-21 15:16:08
LastEditTime: 2020-08-27 19:27:50
FilePath: /Assignment3-1/intention/business.py
Desciption: 建立fasttext 模型， 判断用户输入是否属于业务咨询。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import logging
import sys
import os

import fasttext
import jieba.posseg as pseg
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from config import root_path
from preprocessor import clean, filter_content
import config

#tqdm.pandas()

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class Intention(object):
    def __init__(self,
                 data_path=config.train_path,  # Original data path.
                 sku_path=config.ware_path,  # Sku file path.
                 model_path=None,  # Saved model path.
                 kw_path=None,  # Key word file path.
                 model_train_file=config.business_train,  # Path to save training data for intention.
                 model_test_file=config.business_test):  # Path to save test data for intention.
        self.model_path = model_path
        self.data = pd.read_csv(data_path)

        if model_path and os.path.exists(model_path):
            self.fast = fasttext.load_model(model_path)
        else:
            self.kw = self.build_keyword(sku_path, to_file=kw_path)
            self.data_process(model_train_file)  # Create
            self.fast = self.train(model_train_file, model_test_file)

    def build_keyword(self, sku_path, to_file):
        '''
        @description: 构建业务咨询相关关键词，并保存
        @param {type}
        sku_path： JD sku 文件路径
        to_file： 关键词保存路径
        @return: 关键词list
        '''

        logging.info('Building Keywords.')
        tokens=[]
        
        #get all words with the right POS
        tokens = self.data['custom'].dropna().apply(
            lambda x:[token for token,pos in pseg.cut(x) if pos in ['n','vn','nz']]
        )
        
        #create a set of keywords from the list of tokens
        key_words = set([e for idx, token in enumerate(tokens) for e in token if len(e)>1])
        logging.info('Keywords Built.')
        
        
        sku=[]
        with open(sku_path,'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split('\t')
                sku.extend(line[-1].split('/'))
        key_words = key_words | set(sku)
        logging.info('Sku Words Merged')
        if to_file is not None:
            with open(to_file,'w') as f:
                for i in key_words:
                    f.write(i+"\n")
                    
        return key_words

    def data_process(self, model_data_file):
        '''
        @description: 判断咨询中是否包含业务关键词， 如果包含label为1， 否则为0
                      并处理成fasttext 需要的数据格式
        @param {type}
        model_data_file： 模型训练数据保存路径
        @return:
        '''
        logging.info('Processing data.')
        self.data['is_business'] = self.data['custom'].apply(
            lambda x: 1 if any(kw in x for kw in self.kw) else 0
        )
        with open(model_data_file,'w') as f:
            for index, row in tqdm(self.data.iterrows(),total = self.data.shape[0]):
                output = clean(row['custom'])+"\t__label__"+str(int(row['is_business']))+"\n"
                f.write(output)
                
    def train(self, model_data_file, model_test_file):
        '''
        @description: 读取模型训练数据训练， 并保存
        @param {type}
        model_data_file： 模型训练数据位置
        model_test_file： 模型验证文件位置
        @return: fasttext model
        '''
        logging.info('Training classifier.')
        classifier = fasttext.train_supervised(model_data_file,
                                              label="__label__",
                                              dim=100,
                                              epoch = 5,
                                              lr = 0.1,
                                              wordNgrams =2,
                                              loss="softmax",
                                              thread=5,
                                              verbose=True)
        self.test(classifier, model_test_file)
        classifier.save_model(self.model_path)
        logging.info('Model Saved')
        return classifier

    def test(self, classifier, model_test_file):
        '''
        @description: 验证模型
        @param {type}
        classifier： model
        model_test_file： 测试数据路径
        @return:
        '''
        logging.info('Testing trained model.')
        test_set = pd.read_csv(config.test_path).fillna('')
        test_set['is_business']= test_set['custom'].apply(
            lambda x: 1 if any(kw in x for kw in self.kw) else 0
        )
        with open(model_test_file,'w') as f:
            for index, row in tqdm(self.data.iterrows(),total = self.data.shape[0]):
                output = clean(row['custom'])+"\t__label__"+str(int(row['is_business']))+"\n"
                f.write(output)
        result = classifier.test(model_test_file)
        # F1 score
        print(result[1] * result[2] * 2 / (result[2] + result[1]))

    def predict(self, text):
        '''
        @description: 预测
        @param {type}
        text： 文本
        @return: label, score
        '''
        logging.info('Predicting.')
        label, score = self.fast.predict(clean(filter_content(text)))
        return label, score


if __name__ == "__main__":
    it = Intention(config.train_path,
                 config.ware_path,
                 model_path=config.ft_path,
                 kw_path=config.keyword_path)
    print(it.predict('怎么申请价保呢？'))
    print(it.predict('你好'))