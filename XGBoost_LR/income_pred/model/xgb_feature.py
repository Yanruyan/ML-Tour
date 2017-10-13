#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 构建xgboost特征

from util import dataProcess,xgb_util,parseXgbTree
import xgboost as xgb

# A2中的8个连续特征转换为libSVM格式
dataPath = "E:\\MachineLearning\\data\\classfication\\kdd_census\\census-income.data"
raw_data = dataProcess.read_raw_data(dataPath)
# continousFeatNames = [ "age", "wage_per_hour", "capital_gains", "capital_losses",
#                        "dividends_stocks",  "code_change_msa", "employer_num", "work_weeks_per_year" ]
# A2_raw = raw_data[continousFeatNames]
# A2_labels_raw = raw_data["Y"]
# A2_X = A2_raw.iloc[40000:160000,:]
# A2_y = A2_labels_raw.iloc[40000:160000]
# A2_path = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_xgb_input.txt"
# dataProcess.continousFeatureSVM( A2_X, continousFeatNames, A2_y, A2_path )

##################### xgboost特征 #####################################################################################

# 8个连续特征（libSVM格式）
raw_feat_data = xgb.DMatrix( "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_xgb_input.txt" )

# 载入训练好的xgboost模型
xgb_base = xgb.Booster({'nthread':2})       # 初始化一个booster模型
xgb_base.load_model( "../modelFile/xgb_base_bin" )   # 载入训练好的模型（必须是二进制模型文件）

# 预测（预测方式：每个样本落在树的哪个叶子节点上）
predictByLeaf = xgb_base.predict( data=raw_feat_data, pred_leaf=True )     # 120000行 X 100列的二维矩阵
xgb_util.dumpPredictLeaf2File( predictByLeaf, "../data/predict_by_leaf.txt" ) # dump

# 解析xgboost模型每棵树的叶子节点
xgb_tree_dict = parseXgbTree.parseXgbTreeLeaf("../modelFile/xgb_base_model.txt")  # 用文本格式的模型文件解析
offset_dict = [503]
offset = 503
for treeLeaf in xgb_tree_dict:
    offset = offset + len( treeLeaf )
    offset_dict.append( offset )

########################## 类别型特征 #################################################################################

# 33个类别型特征
catFeatures = [("worke_class", 9), ("industry_code", 52), ("occup_code", 47), ("education", 17), ("enroll_edu", 3),
               ("marital_stat", 7), ("major_industry_code", 24), ("major_occup_code", 15), ("race", 5),
               ("hispanic_origin", 10), ("sex", 2), ("member_labor_union", 3), ("reason_unemployment", 6),
               ("employment_stat", 8), ("tax_filer_stat", 6), ("region_previous_residence", 6),
               ("state_previous_residence", 51), ("household_family_stat", 38), ("household_summary", 8),
               ("code_change_reg", 9), ("code_move_reg", 10), ("dump", 10),
               ("in_house_1year_ago", 3), ("migration_prev_res", 4), ("family_members_under_18", 5),
               ("father_country", 43), ("mother_country", 43), ("self_country", 43), ("citizenship", 5),
               ("own_business", 3), ("fill_questionnaire_veteran_admin", 3), ("veterans_benefits", 3), ("year",2)
               ]

# 获取每个特征的类别信息
featCateDict = {}
featCateCountDict = {}
for feat in catFeatures:
    tmp_cates,tmp_catCount = dataProcess.getFeatCateInfo( raw_data, feat[0] )
    featCateDict[feat[0]] = tmp_cates
    featCateCountDict[feat[0]] = tmp_catCount

# 离散特征数据
catFeatNames = []
for feat in catFeatures:
    catFeatNames.append( feat[0] )
lr_df = raw_data[catFeatNames]
labels = raw_data["Y"]
nomial_x = lr_df.iloc[40000:160000,:]
nomial_y = labels.iloc[40000:160000]

######################## 特征合并：类别特征+xgboost特征 ###############################################################

combFeatPath = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_all_feature_svm.txt"
dataProcess.mergeXgbFeature( nomial_x, catFeatNames, featCateDict, featCateCountDict, nomial_y,
                             predictByLeaf, xgb_tree_dict, offset_dict, combFeatPath )

