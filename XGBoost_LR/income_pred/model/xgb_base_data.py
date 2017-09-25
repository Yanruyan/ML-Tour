#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 8个连续特征训练XGBoost模型的训练数据生成（libSVM格式）

from util import dataProcess

# 读取训练数据
dataPath = "E:\\MachineLearning\\data\\classfication\\kdd_census\\census-income.data"
raw_data = dataProcess.read_raw_data(dataPath)

# 8个连续特征
continousFeatNames = [ "age", "wage_per_hour", "capital_gains", "capital_losses",
                       "dividends_stocks",  "code_change_msa", "employer_num", "work_weeks_per_year" ]
xgb_df = raw_data[continousFeatNames]
labels = raw_data["Y"]

# 连续特征做one-hot处理并转化为libSVM格式
# train set
xgb_train_x = xgb_df.iloc[0:30000,:]
xgb_train_y = labels.iloc[0:30000]
train_path = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A1_train_svm_data.txt"
dataProcess.continousFeatureSVM( xgb_train_x, continousFeatNames, xgb_train_y, train_path )
# test set
xgb_test_x = xgb_df.iloc[30000:40000,:]
xgb_test_y = labels.iloc[30000:40000]
test_path = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A1_test_svm_data.txt"
dataProcess.continousFeatureSVM( xgb_test_x, continousFeatNames, xgb_test_y, test_path )

