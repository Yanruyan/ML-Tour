#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 33个离散特征训练LR模型的数据

from util import dataProcess
from sklearn.preprocessing import LabelEncoder

# 读取训练数据
dataPath = "E:\\MachineLearning\\data\\classfication\\kdd_census\\census-income.data"
raw_data = dataProcess.read_raw_data(dataPath)

# 先用33离散特征训练LR
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

# 离散特征做one-hot处理并转化为libSVM格式
# train set.
lr_train_x = lr_df.iloc[40000:160000,:]
lr_train_y = labels.iloc[40000:160000]
train_path = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_train_svm_data.txt"
dataProcess.featureOneHotSVM( lr_train_x, catFeatNames, featCateDict, featCateCountDict, lr_train_y, train_path )
# test set.
lr_test_x = lr_df.iloc[160000:199523,:]
lr_test_y = labels.iloc[160000:199523]
test_path = "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_test_svm_data.txt"
dataProcess.featureOneHotSVM( lr_test_x, catFeatNames, featCateDict, featCateCountDict, lr_test_y, test_path )






