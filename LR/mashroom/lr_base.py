#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 第1版LR模型

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
dataPath = "E:\\MachineLearning\\data\\mushroom\\mushroom.csv"
featNames = [ "Y", "capShape", "capSurface", "capColor", "bruises", "odor", "gillAttach", "gillSpace",
              "gillSize", "gillColor", "stalkShape", "stalkRoot", "stalkSurfAbove", "stalkSurfBelow",
              "stalkColorAbove", "stalkColorBelow", "veilType", "veilColor", "ringNum", "ringType",
              "sporeColor", "population", "habitat" ]
raw_data = pd.read_csv( dataPath, names = featNames )

# 类别标签Y处理
le = LabelEncoder()
raw_data["Y"] = le.fit_transform( raw_data["Y"] )  # 用LabelEncoder拟合Y（fit)、再得到encoder后的新的
                                                    # 类别标签Y列表（transform)，2步一起做。

# 类别特征预处理（One-Hot编码）
catFeatures = [ ("capShape",6), ("capSurface",4), ("capColor",10), ("bruises",2), ("odor",9), ("gillAttach",4),
                 ("gillSpace",3), ("gillSize",2), ("gillColor",12), ("stalkShape",2), ("stalkRoot",7),
                 ("stalkSurfAbove",4), ("stalkSurfBelow",4), ("stalkColorAbove",9), ("stalkColorBelow",9),
                 ("veilType",2), ("veilColor",4), ("ringNum",3), ("ringType",8), ("sporeColor",9),
                 ("population",6), ("habitat",7) ]
for feat in catFeatures:
    raw_data = util.featureOneHot( raw_data, feat[0], feat[1] )

# 训练集&测试集
y_train = raw_data.iloc[:,0][0:6500]
x_train = raw_data.iloc[:,1:118][0:6500]
y_test_real = raw_data.iloc[:,0][6500:8124]
x_test = raw_data.iloc[:,1:118][6500:8124]

# 训练集上训练LR模型（默认超参数）
lr_base = LogisticRegression( penalty='l1', C=1.0, fit_intercept=True, intercept_scaling=1,
                              solver='liblinear', max_iter=100, multi_class='ovr', n_jobs=1 )
lr_base.fit( X=x_train, y=y_train )
coef_lr_base = lr_base.coef_.ravel()

# 测试集上模型评价
y_test_pred = lr_base.predict_proba( x_test )  # probability
fpr, tpr, threshold = roc_curve( y_test_real, y_test_pred[:, 1] )
# AUC
auc_base = auc( fpr, tpr )
print "lr_base AUC:", auc_base
# ROC curve
plt.plot( fpr, tpr, lw=1, alpha=0.3, label='ROC-Curve' )
plt.show()

