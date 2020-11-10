#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:16
LastEditTime: 2020-09-30 05:02:34
FilePath: /Assignment3-3_solution/generative/predict.py
Desciption: Predict using BERT seq2seq model.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import sys
import os

import torch

sys.path.append('..')
from config import is_cuda, root_path
from generative.bert_model import BertConfig
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import load_chinese_base_vocab




class bertSeq2Seq(object):
    def __init__(self, model_path, is_cuda):
        self.word2idx = load_chinese_base_vocab()
        self.config = BertConfig(len(self.word2idx))
        self.bert_seq2seq = Seq2SeqModel(self.config)
        self.is_cuda = is_cuda
        if is_cuda:
            device = torch.device("cuda")
            self.bert_seq2seq.load_state_dict(torch.load(model_path))
            self.bert_seq2seq.to(device)
        else:
            checkpoint = torch.load(model_path,
                                    map_location=torch.device("cpu"))
            self.bert_seq2seq.load_state_dict(checkpoint)
        # 加载state dict参数
        self.bert_seq2seq.eval()

    def generate(self, text, k=5):
        result = self.bert_seq2seq.generate(text,
                                            beam_size=k,
                                            is_cuda=self.is_cuda)
        return result


if __name__ == "__main__":
    bs = bertSeq2Seq(os.path.join(root_path, 'model/generative/bert.model.epoch.29'), is_cuda)
    text = '谢 谢 学 姐'
    print(bs.generate(text, k=5))
