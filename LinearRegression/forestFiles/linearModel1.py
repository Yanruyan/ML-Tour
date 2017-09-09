#!/usr/bin/python
# -*- coding: UTF-8 -*-
# ForestFires的线性回归建模：第1版

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from sklearn import metrics

# 从CSV文件中读取数据
dataPath = "E:\\MachineLearning\\data\\forest_fires\\forestfires.csv"
featNames = [ "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area" ]
raw_data = pd.read_csv( dataPath, names = featNames, header=0 )

# 因变量area做log1p变换
area_freq = raw_data["area"].value_counts()   # 大部分都是0，非0值很少，说明area不服从正态分布
raw_data["area"] = np.log1p( raw_data["area"].values )    # 重新设置列“area”的值（对area做ln(x+1)变换，转换为正态分布）

# 先删除离散特征
del raw_data["month"]
del raw_data["day"]

# 训练集、测试集
train_data = raw_data.iloc[0:450,:]
test_data = raw_data.iloc[450:517,:]

# 在训练集上拟合线性回归模型
x_train = train_data.iloc[:,0:10]
y_train = train_data.iloc[:,10]
linearModel = LinearRegression()
linearModel.fit( x_train, y_train )
print "coef:",linearModel.coef_
print "intercept:",linearModel.intercept_
print "residues:",linearModel.residues_

# 在测试集上对模型进行评价（RMSE）
# 1、预测测试集
x_test = test_data.iloc[:,0:10]
y_test_real = map( lambda x:(math.exp(x)-1), test_data["area"].values )  # 转换后的真实的area值
y_test_pred_log = linearModel.predict(x_test)
y_test_pred = map( lambda x:(math.exp(x)-1), y_test_pred_log )  # 转换后的预测的area值
# 2、计算RMSE
mse = metrics.mean_squared_error(y_test_real,y_test_pred)
rmse = np.sqrt( metrics.mean_squared_error(y_test_real,y_test_pred) )
print "mse:",mse                   # mse: 1574.9142468
print "rmse:",rmse                 # rmse: 39.6851892625（误差非常大，继续改进模型）

