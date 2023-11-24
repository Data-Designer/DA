#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 15:17
# @Author  : Jack Zhao
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc: 一些配置参数


class TrainConfig():
    """Configs"""
    DATA = "clef"
    GPU_USED = True
    MEMENTUM = 1
    LR = 1e-3 
    GAMMA = 0.0003
    DECAY = 0.75
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 32
    SHUFFLE = True
    START = 1 #
    EPOCHS = 120 
    RESNET = 50
    PSEUINTERVEAL = 2000 
    DISLAYER = 2 # Classifier Layer Num
    CLASSNUM = 12 # Class Number
    GRADNUM = 12 
    NEARK = 5
    OPTIM = "momentum"
    GUPDATE = 4 
    TRAPATH = "/root/autodl-tmp/dfs/data/DA/data/clef/c"
    VALPATH = "/root/autodl-tmp/dfs/data/DA/data/clef/i"
    LOGFILE = "/root/autodl-tmp/dfs/data/DA/log/clef/log1/result_new.csv" 
    TLOGFILE = "/root/autodl-tmp/dfs/data/DA/log/clef/log1/run"
    WEIGHTS = "/root/autodl-tmp/dfs/data/DA/log/clef/log1/checkpoint/"
    CASEFILE = "/root/autodl-tmp/dfs/data/DA/log/clef/log1/case.csv"
    TSNE = "/root/autodl-tmp/dfs/data/DA/log/clef/log1/last-tsne.pdf"
    UNCERTAINTY = 0.85 
    CLOSS = 0.05 
    DISC = 1
    BOTTLE = 1024
    ENTROPY = 0.1
    GMLOSS = 0.01
    USE_BOTTLE= False
    SEED = 100 
    LOG_STEP_FREQ = 100 



def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))






TrainConfig.parse = parse
opt = TrainConfig() 