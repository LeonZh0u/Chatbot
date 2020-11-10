#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:16
LastEditTime: 2020-10-16 17:34:22
FilePath: /Assignment3-3_solution/generative/train_distil.py
Desciption: Perform knowledge distillation for compressing the BERT model.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import os
import random
import sys
from functools import partial
import csv

import numpy as np
import pandas as pd
import torch
from textbrewer import DistillationConfig, GeneralDistiller, TrainingConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
sys.path.append('..')
from config import root_path
from generative import config_distil
from generative.bert_model import BertConfig
from generative.optimizer import BERTAdam
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.tools import create_logger, divide_parameters


def read_corpus(data_path):
    df = pd.read_csv(data_path,
                     sep='\t',
                     header=None,
                     names=['src', 'tgt'],
                     quoting=csv.QUOTE_NONE
                    ).dropna()
    sents_src = []
    sents_tgt = []
    for index, row in df.iterrows():
        query = row["src"]
        answer = row["tgt"]
        sents_src.append(query)
        sents_tgt.append(answer)
    return sents_src, sents_tgt


# 自定义dataset
class SelfDataset(Dataset):
    """
    针对数据集，定义一个相关的取数据的方式
    """
    def __init__(self, path, max_length):
        # 一般init函数是加载所有数据
        super(SelfDataset, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = read_corpus(path)
        self.word2idx = load_chinese_base_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

        self.max_length = max_length

    def __getitem__(self, i):
        # 得到单个数据

        src = self.sents_src[i] if len(
            self.sents_src[i]
        ) < self.max_length else self.sents_src[i][:self.max_length]
        tgt = self.sents_tgt[i] if len(
            self.sents_tgt[i]
        ) < self.max_length else self.sents_tgt[i][:self.max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice).to(args.device)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


def main():

    config_distil.parse()
    global args
    args = config_distil.args
    global logger
    logger = create_logger(args.log_file)

    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")

    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    logger.info('加载字典')
    word2idx = load_chinese_base_vocab()
    # 判断是否有可用GPU
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and args.is_cuda else "cpu")


    logger.info('using device:{}'.format(args.device))
    # 定义模型超参数
    bertconfig_T = BertConfig(vocab_size=len(word2idx))
    bertconfig_S = BertConfig(vocab_size=len(word2idx), num_hidden_layers=3)
    logger.info('初始化BERT模型')
    bert_model_T = Seq2SeqModel(config=bertconfig_T)
    bert_model_S = Seq2SeqModel(config=bertconfig_S)
    logger.info('加载Teacher模型～')
    load_model(bert_model_T, args.tuned_checkpoint_T)
    logger.info('将模型发送到计算设备(GPU或CPU)')
    bert_model_T.to(args.device)
    bert_model_T.eval()

    logger.info('加载Student模型～')
    if args.load_model_type == 'bert':
        load_model(bert_model_S, args.init_checkpoint_S)
    else:
        logger.info(" Student Model is randomly initialized.")
    logger.info('将模型发送到计算设备(GPU或CPU)')
    bert_model_S.to(args.device)
    # 声明自定义的数据加载器

    logger.info('加载训练数据')
    train = SelfDataset(args.train_path, args.max_length)
    trainloader = DataLoader(train,
                             batch_size=args.train_batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)

    if args.do_train:

        logger.info(' 声明需要优化的参数')
        num_train_steps = int(
            len(trainloader) / args.train_batch_size) * args.num_train_epochs
        optim_parameters = list(bert_model_S.named_parameters())
        all_trainable_params = divide_parameters(optim_parameters,
                                                 lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d",
                    len(all_trainable_params))
        optimizer = BERTAdam(all_trainable_params,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps,
                             schedule=args.schedule,
                             s_opt1=args.s_opt1,
                             s_opt2=args.s_opt2,
                             s_opt3=args.s_opt3)

        train_config = TrainingConfig(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ckpt_frequency=args.ckpt_frequency,
            log_dir=args.output_dir,
            output_dir=args.output_dir,
            device=args.device)

        from generative.matches import matches
        intermediate_matches = None

        
        if isinstance(args.matches, (list, tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        intermediate_matches = []
        for match in args.matches:
            intermediate_matches += matches[match]
        
        
        logger.info(f"{intermediate_matches}")
        distill_config = DistillationConfig(
            temperature=args.temperature,
            intermediate_matches=intermediate_matches)

        def BertForS2SSimpleAdaptor(batch, model_outputs):
            return {'hidden': model_outputs[0], 'logits': model_outputs[1], 'loss': model_outputs[2], 'attention': model_outputs[3]}

        
        adaptor_T = partial(BertForS2SSimpleAdaptor)
        adaptor_S = partial(BertForS2SSimpleAdaptor)

        distiller = GeneralDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=bert_model_T,
                                     model_S=bert_model_S,
                                     adaptor_T=adaptor_T,
                                     adaptor_S=adaptor_S)
        callback_func = partial(predict, data_path=args.dev_path, args=args)
        logger.info('Start distillation.')
        with distiller:
            distiller.train(optimizer,
                            scheduler=None,
                            dataloader=trainloader,
                            num_epochs=args.num_train_epochs,
                            callback=None)

    if not args.do_train and args.do_predict:
        res = predict(bert_model_S, args.test_path, step=0, args=args)
        print(res)


def load_model(model, pretrain_model_path):

    checkpoint = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
    # 模型刚开始训练的时候, 需要载入预训练的BERT

    checkpoint = {
        k[5:]: v
        for k, v in checkpoint.items() if k[:4] == "bert" and "pooler" not in k
    }
    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.empty_cache()
    logger.info("{} loaded!".format(pretrain_model_path))


def predict(model, data_path, step, args):
    logger.info('加载测试数据')
    dev = SelfDataset(data_path, args.max_length)
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("Batch size = %d", args.predict_batch_size)

    devloader = DataLoader(dev,
                           batch_size=args.predict_batch_size,
                           shuffle=True,
                           collate_fn=collate_fn)

    model.eval()
    all_results = []
    logger.info("Start evaluating")

    for batch_idx, data in enumerate(
            tqdm(devloader, desc="Evaluating", disable=None)):
        
        token_ids, token_type_ids, target_ids = data
        token_ids = token_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        target_ids = target_ids.to(args.device)
        with torch.no_grad():
            predictions, loss = model(token_ids,
                                      token_type_ids,
                                      labels=target_ids)
            all_results.append(predictions)
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
                logger.info("evaluate batch {} ,loss {}".format(batch_idx, loss))
        
    logger.info("finishing evaluating")


if __name__ == "__main__":
    main()
