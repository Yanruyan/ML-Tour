#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 8个连续特征训练xgboost

import xgboost as xgb
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

# 读取数据（标准libsvm格式）
train_data = xgb.DMatrix( "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A1_train_svm_data.txt" )
test_data = xgb.DMatrix( "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A1_test_svm_data.txt" )

# 超参数设置
# 1、xgboost模型参数（包括：通用参数、booster参数、训练学习参数）
param = {
    "eta":0.03,
    "max_depth":5,
    "silent":1,
    "objective":"binary:logistic",
    "eval_metric":"auc"
}
# 2、boosting次数，相当于构建50棵树
boost_iter_round = 100
# 3、看板，每次迭代都可以在控制台打印出训练集与测试集的指标（如error、auc等）
watchlist = [(test_data, 'test'), (train_data, 'train')]

# 训练模型
xgb_base = xgb.train( params=param, dtrain=train_data, num_boost_round=boost_iter_round, evals=watchlist )

# 模型评价
# 1、测试集上预测
y_test_pred = xgb_base.predict(test_data)
y_test_real = test_data.get_label()
# 2、测试集上错误率
error_count = sum( y_test_real != (y_test_pred>0.5) )
error_rate = float(error_count) / len(y_test_pred)
print "error_rate:",error_rate     # error_rate : 0.0513350445015
# 3、测试集上ROC曲线、AUC
# auc
fpr, tpr, threshold = roc_curve( y_true=y_test_real, y_score=y_test_pred )
auc_base = auc( fpr, tpr )
print "auc_base:",auc_base    # AUC : 0.91536472245
# roc
plt.plot( fpr, tpr, lw=1, alpha=0.3, label='ROC-Curve' )
plt.show()

# 保存xgb_base模型
# 文本格式的模型文件，便于解析
xgb_base.dump_model( fout="E:\\MachineLearning\\ml\\XGBoost_LR\\income_pred\\modelFile\\xgb_base_model.txt" )
# 二进制格式的模型文件，后面可以直接加载模型
xgb_base.save_model( fname="E:\\MachineLearning\\ml\\XGBoost_LR\\income_pred\\modelFile\\xgb_base_bin" )

