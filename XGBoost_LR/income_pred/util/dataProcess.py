#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 数据处理&划分数据集
# 数据集说明：用户的一些特征，预测用户的收入是<50000还是>50000

import pandas as pd
from sklearn.datasets import dump_svmlight_file
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#######################################################################################################################
# 读取数据
def read_raw_data(dataPath):
    featNames = [
        "age", "worke_class", "industry_code", "occup_code", "education", "wage_per_hour", "enroll_edu",
        "marital_stat", "major_industry_code", "major_occup_code", "race", "hispanic_origin", "sex",
        "member_labor_union", "reason_unemployment", "employment_stat", "capital_gains", "capital_losses",
        "dividends_stocks", "tax_filer_stat", "region_previous_residence", "state_previous_residence",
        "household_family_stat", "household_summary", "code_change_msa", "code_change_reg", "code_move_reg",
        "dump","in_house_1year_ago", "migration_prev_res", "employer_num", "family_members_under_18",
        "father_country","mother_country", "self_country", "citizenship", "own_business",
        "fill_questionnaire_veteran_admin","veterans_benefits", "work_weeks_per_year", "year", "Y"
    ]
    raw_data = pd.read_csv(dataPath, names=featNames,index_col=False)  # index_col=False表示：第1列是特征，不是index名
    # 分类label（- 50000.、50000+.）转为0/1值
    le = LabelEncoder()
    raw_data["Y"] = le.fit_transform(raw_data["Y"])
    return raw_data

#######################################################################################################################
# 获取训练xgboost模型的数据
def getXgbTrainData(raw_data):
    A1 = raw_data.iloc[0:40000, :]
    # 7个连续特征
    X_A1 = A1[ ["age", "wage_per_hour", "capital_gains", "capital_losses",
                "dividends_stocks", "employer_num", "work_weeks_per_year"] ]
    Y_A1 = A1["Y"]
    # 训练集
    X_A1_train = X_A1.iloc[0:30000,:]
    Y_A1_train = Y_A1.iloc[0:30000]
    dump_svmlight_file(X=X_A1_train, y=Y_A1_train, f="E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A1_train_svm_data",
                       zero_based=True, multilabel=False)
    # 测试集
    X_A1_test = X_A1.iloc[30000:40000,:]
    Y_A1_test = Y_A1.iloc[30000:40000]
    dump_svmlight_file(X=X_A1_test, y=Y_A1_test,
                       f="E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A1_test_svm_data",
                       zero_based=True, multilabel=False)

#######################################################################################################################
# 单特征One-Hote编码，并加入到原来的样本数据dataFrame中
# df：样本数据dataFrame
# sigfeatName：特征名
# labelName：特征类别数
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



def div_data(raw_data):
    # A2：用原始特征+xgboost组合特征训练LR的数据集
    A2 = raw_data.iloc[40000:160000, :]
    featNames = [
        "age", "worke_class", "industry_code", "occup_code", "education", "wage_per_hour", "enroll_edu",
        "marital_stat", "major_industry_code", "major_occup_code", "race", "hispanic_origin", "sex",
        "member_labor_union", "reason_unemployment", "employment_stat", "capital_gains", "capital_losses",
        "dividends_stocks", "tax_filer_stat", "region_previous_residence", "state_previous_residence",
        "household_family_stat", "household_summary", "code_change_msa", "code_change_reg", "code_move_reg",
        "in_house_1year_ago", "migration_prev_res", "employer_num", "family_members_under_18", "father_country",
        "mother_country", "self_country", "citizenship", "own_business", "fill_questionnaire_veteran_admin",
        "veterans_benefits", "work_weeks_per_year", "year", "Y"
    ]
    A2.to_csv( "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A2_data",
               columns=featNames, index=False, header=False )
    # A3:验证LR模型的测试集
    A3 = raw_data.iloc[160000:199523, :]
    A3.to_csv( "E:\\MachineLearning\\data\\classfication\\kdd_census\\data\\A3_data",
               columns=featNames, index=False, header=False )


if __name__=="__main__":
    dataPath = "E:\\MachineLearning\\data\\classfication\\kdd_census\\census-income.data"
    raw_data = read_raw_data(dataPath)
    getXgbTrainData(raw_data)


























