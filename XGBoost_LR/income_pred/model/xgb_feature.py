#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 构建xgboost特征

from util import dataProcess,xgb_util,parseXgbTree
import xgboost as xgb

# A2中的8个连续特征转换为libSVM格式
# dataPath = "E:\\MachineLearning\\data\\classfication\\kdd_census\\census-income.data"
# raw_data = dataProcess.read_raw_data(dataPath)
# continousFeatNames = [ "age", "wage_per_hour", "capital_gains", "capital_losses",
#                        "dividends_stocks",  "code_change_msa", "employer_num", "work_weeks_per_year" ]
# A2_raw = raw_data[continousFeatNames]
# A2_labels_raw = raw_data["Y"]
# A2_X = A2_raw.iloc[40000:160000,:]
# A2_y = A2_labels_raw.iloc[40000:160000]
# A2_path = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_xgb_svm_data.txt"
# dataProcess.continousFeatureSVM( A2_X, continousFeatNames, A2_y, A2_path )

# 读取连续特征数据（libSVM格式）
raw_feat_data = xgb.DMatrix( "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_xgb_svm_data.txt" )

# 载入训练好的xgboost模型
xgb_base = xgb.Booster({'nthread':2})       # 初始化一个booster（模型）
xgb_base.load_model( "../modelFile/xgb_base_bin" )   # 载入训练好的模型（必须是二进制模型文件）

# 预测（预测方式：每个样本落在树的哪个叶子节点上）
predictByLeaf = xgb_base.predict( data=raw_feat_data, pred_leaf=True )     # 120000行 X 100列的二维矩阵
xgb_util.dumpPredictLeaf2File( predictByLeaf, "../data/predict_by_leaf.txt" ) # dump

# 解析xgboost模型每棵树的叶子节点
xgb_tree_dict = parseXgbTree.parseXgbTreeLeaf("../modelFile/xgb_base_model.txt")  # 用文本格式的模型文件解析


































