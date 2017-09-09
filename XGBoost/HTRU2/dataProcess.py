#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 数据处理：HTRU2的数据处理、划分训练集与测试集、数据转libsvm格式。

import pandas as pd
from sklearn.datasets import dump_svmlight_file

# 读取数据
dataPath = "E:\\MachineLearning\\data\\classfication\\HTRU2\\HTRU_2.csv"
featNames = [ "profileMean", "profileStd", "profileSke", "profileKurt", "dmMean",
              "dmStd", "dmSke", "dmKurt", "Y" ]
raw_data = pd.read_csv( dataPath, names=featNames )

# # Y移动到第1列作为标签
# # Y dataFrame
# y_data = raw_data["Y"]
# y_df = pd.DataFrame(y_data)
# # raw_data delete "Y"
# del raw_data["Y"]
# # merge Y-dataFrame and raw_data dataFrame
# raw_data = y_df.join(raw_data)

# 划分训练集、测试集
train_data = raw_data.iloc[0:14320,:]
test_data = raw_data.iloc[14320:17898,:]

# 训练数据转为libsvm格式
X_train = train_data.iloc[:,0:8]
Y_train = train_data["Y"]
dump_svmlight_file( X=X_train, y=Y_train,f="E:\\MachineLearning\\data\\classfication\\HTRU2\\train_svm_data",
                    zero_based=True, multilabel=False )

# 测试数据转为libsvm格式
X_test = test_data.iloc[:,0:8]
Y_test = test_data["Y"]
dump_svmlight_file( X=X_test, y=Y_test, f="E:\\MachineLearning\\data\\classfication\\HTRU2\\test_svm_data",
                    zero_based=True, multilabel=False )
