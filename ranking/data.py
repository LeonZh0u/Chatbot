#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-11 11:44:54
LastEditTime: 2020-09-11 14:16:34
FilePath: /Assignment3-2_solution/ranking/data.py
Desciption: Data processing for ranking module.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import pandas as pd
import torch
from torch.utils.data import Dataset
import csv


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """
    def __init__(self, bert_tokenizer, file, max_char_len=103):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels \
            = self.get_input(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[
            idx], self.labels[idx]
    def is_int(self, val):
        try:
            num = int(val)
        except ValueError:
            return False
        return True
    # 获取文本与标签
    def get_input(self, file):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        df = pd.read_csv(file)
        df.dropna()
#         df = df[df['question1'].isin(
#             np.random.choice(df['question1'].unique(),
#                              int(0.3 * df['question1'].unique().shape[0])))]
        list1=[]
        for i,lab in enumerate(df['label']):
            if not self.is_int(str(lab)):
                print(lab,i)
                list1.append(i)
        print(file)
        df = df.drop(df.index[list1])
        df["question1"] = df["question1"].apply(lambda x: "".join(x.split()))
        df["question2"] = df["question2"].apply(lambda x: "".join(x.split()))
        labels = df['label'].astype('int8').values

        # 切词
        tokens_seq_1 = list(
            map(self.bert_tokenizer.tokenize, df['question1'].values))
        tokens_seq_2 = list(
            map(self.bert_tokenizer.tokenize, df['question2'].values))
        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(
            torch.long), torch.Tensor(seq_masks).type(
                torch.long), torch.Tensor(seq_segments).type(
                    torch.long), torch.Tensor(labels).type(torch.long)

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参:
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分

        """
        # 对超长序列进行截断
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3) // 2]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) +
                             2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment
