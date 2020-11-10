#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-30 05:03:31
LastEditTime: 2020-10-16 17:38:08
FilePath: /Assignment3-3_solution/generative/matches.py
Desciption: Define layer matches and losses for KD.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
L3_attention_mse=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_mse", "weight":1},
                  {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_mse", "weight":1}]

L3_attention_ce=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_ce", "weight":1},
                 {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_ce", "weight":1}]

L3_attention_mse_sum=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_mse_sum", "weight":1},
                      {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_mse_sum", "weight":1}]

L3_attention_ce_mean=[{"layer_T":4,  "layer_S":1, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":8,  "layer_S":2, "feature":"attention", "loss":"attention_ce_mean", "weight":1},
                      {"layer_T":12, "layer_S":3, "feature":"attention", "loss":"attention_ce_mean", "weight":1}]

L3_hidden_smmd=[{"layer_T":[0,0],  "layer_S":[0,0], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[4,4],  "layer_S":[1,1], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[8,8],  "layer_S":[2,2], "feature":"hidden", "loss":"mmd", "weight":1},
                {"layer_T":[12,12],"layer_S":[3,3], "feature":"hidden", "loss":"mmd", "weight":1}]

L3n_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]},
                {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",384,768]}]

L3_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1},
               {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1}]

L3l_hidden_mse=[{"layer_T":0, "layer_S":0, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]},
                {"layer_T":4, "layer_S":1, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]},
                {"layer_T":8, "layer_S":2, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]},
                {"layer_T":12,"layer_S":3, "feature":"hidden", "loss":"hidden_mse", "weight":1, "proj":["linear",1024,768]}]
#######################L4################
L4_attention_mse = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}]

L4_attention_ce = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}]

L4_attention_mse_sum = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}]

L4_attention_ce_mean = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}]

L4_hidden_smmd = [{
    "layer_T": [0, 0],
    "layer_S": [0, 0],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [3, 3],
    "layer_S": [1, 1],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [6, 6],
    "layer_S": [2, 2],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [9, 9],
    "layer_S": [3, 3],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [12, 12],
    "layer_S": [4, 4],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}]

L4t_hidden_mse = [{
    "layer_T": 0,
    "layer_S": 0,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 3,
    "layer_S": 1,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}]

matches = {
    'L3_attention_mse': L3_attention_mse,
    'L3_attention_mse_sum': L3_attention_mse_sum,
    'L3_attention_ce': L3_attention_ce,
    'L3_attention_ce_mean': L3_attention_ce_mean,
    'L3n_hidden_mse': L3n_hidden_mse,
    'L3_hidden_smmd': L3_hidden_smmd,
    'L3_hidden_mse': L3_hidden_mse,
    'L3l_hidden_mse': L3l_hidden_mse,
    'L4_attention_mse': L4_attention_mse,
    'L4_attention_mse_sum': L4_attention_mse_sum,
    'L4_attention_ce': L4_attention_ce,
    'L4_attention_ce_mean': L4_attention_ce_mean,
    'L4t_hidden_mse': L4t_hidden_mse,
    'L4_hidden_smmd': L4_hidden_smmd,
}
