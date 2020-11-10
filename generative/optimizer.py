#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:16
LastEditTime: 2020-09-30 04:58:08
FilePath: /Assignment3-3_solution/generative/optimizer.py
Desciption: Define the BERTAdam optimizer.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def warmup_linear_release(x,
                          warmup=0.002,
                          opt1=1,
                          opt2=0,
                          opt3=1):  # opt1:release_start  opt2:release_level
    if x < warmup:
        return x / warmup
    if x < opt1:
        return 1 - (1 - opt2) / opt1 * x
    return opt2


def warmup_step(x,
                warmup=0.002,
                opt1=1,
                opt2=0,
                opt3=1):  # opt1:step1 opt2:step2 opt3:decay
    if x < warmup:
        return x / warmup
    if x < opt1:
        return 1.
    if x < opt2:
        return opt3
    return opt3 * opt3


def slanted_triangular(x, warmup=0.1, opt1=1, opt2=0, opt3=1):  # opt1:ratio
    if x < warmup:
        p = x / warmup
    else:
        p = 1 - (x - warmup) / (1 - warmup)
    return (1 + p * (opt1 - 1)) / opt1


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
    'warmup_linear_release': warmup_linear_release,
    'warmup_step': warmup_step,
    'slanted_triangular': slanted_triangular
}


class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self,
                 params,
                 lr,
                 warmup=-1,
                 t_total=-1,
                 schedule='warmup_linear_release',
                 s_opt1=1,
                 s_opt2=0,
                 s_opt3=1,
                 b1=0.9,
                 b2=0.999,
                 e=1e-6,
                 weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(
                    warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(
                    b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(
                    b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))

        defaults = dict(lr=lr,
                        schedule=schedule,
                        warmup=warmup,
                        t_total=t_total,
                        b1=b1,
                        b2=b2,
                        e=e,
                        weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm,
                        opt1=s_opt1,
                        opt2=s_opt2,
                        opt3=s_opt3)

        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr


    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step'] / group['t_total'], group['warmup'],
                        group['opt1'], group['opt2'], group['opt3'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss
