#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 第1版XGBoost模型：默认超参数

import xgboost as xgb
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

# 读取数据
train_data = xgb.DMatrix( "E:\\MachineLearning\\data\\classfication\\HTRU2\\train_svm_data" )
test_data = xgb.DMatrix( "E:\\MachineLearning\\data\\classfication\\HTRU2\\test_svm_data" )

# 超参数设置
# 1、xgboost模型参数（包括：通用参数、booster参数、训练学习参数）
param = {
    "eta":0.05,
    "max_depth":5,
    "silent":1,
    "objective":"binary:logistic",
    "eval_metric":"auc"
}
# 2、boosting次数，相当于构建50棵树
boost_iter_round = 50
# 3、看板，每次迭代都可以在控制台打印出训练集与测试集的指标（如error、auc等）
watchlist = [(test_data, 'eval'), (train_data, 'train')]

# 训练模型
xgb_base = xgb.train( params=param, dtrain=train_data, num_boost_round=boost_iter_round, evals=watchlist )

# 模型评价
# 1、测试集上预测
y_test_pred = xgb_base.predict(test_data)
y_test_real = test_data.get_label()
print "y_test_pred:",y_test_pred
print "y_test_real:",y_test_real
# 2、测试集上错误率
error_count = sum( y_test_real != (y_test_pred>0.5) )
error_rate = float(error_count) / len(y_test_pred)
print "error_rate:",error_rate     # 0.01089
# 3、测试集上ROC曲线、AUC
# auc
fpr, tpr, threshold = roc_curve( y_true=y_test_real, y_score=y_test_pred )
auc_base = auc( fpr, tpr )
print "auc_base:",auc_base
# roc
plt.plot( fpr, tpr, lw=1, alpha=0.3, label='ROC-Curve' )
plt.show()
