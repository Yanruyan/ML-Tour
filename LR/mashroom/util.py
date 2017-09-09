#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 一些数据处理的工具

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#######################################################################
# 单特征One-Hote编码，并加入到原来的样本数据dataFrame中
# df：样本数据dataFrame
# sigfeatName：特征名
# labelName：特征类别数
#
def featureOneHot( df, sigfeatName, labelNum ):
    # actual label count.
    feat_freq = df[sigfeatName].value_counts()
    labelCount = labelNum
    if( len(feat_freq) < labelNum ):
        labelCount = len(feat_freq)
    # dummy feature names.
    dummyFeatNames = []
    idx = 0
    while (idx < labelCount):
        featName = "%s_%d" % (sigfeatName, idx)
        dummyFeatNames.append(featName)
        idx += 1
    # category feature one-hot encoder.
    le = LabelEncoder()
    feat_labels = le.fit_transform( df[sigfeatName] )
    ohEnc = OneHotEncoder( sparse=False )
    feat_oneHot = ohEnc.fit_transform( feat_labels.reshape(-1,1) )
    feat_df = pd.DataFrame( data=feat_oneHot, columns=dummyFeatNames )
    # dummy features merged to dataFrame.
    df = df.join(feat_df)
    # delete raw feature
    del df[sigfeatName]
    return df
