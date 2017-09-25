#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 解析xgboost的树的叶子节点

def parseXgbTreeLeaf(filePath):
    fs = open(filePath,'r')
    lines = fs.readlines()
    # 解析xgboost模型文本，得到每棵树的叶子节点编号[L1,L2,...,Ln]
    # xgboost的m棵树，得到m个数组构成的二维数组treeMap：
    # [
    #   [3,4,5,6],
    #   [7,8,9,10,15,16,12,13,14],
    #   ...
    # ]
    treeMap = []
    booster = []
    for line in lines:
        if "booster[0]" in line:
            booster=[]
        elif "booster" in line:
            treeMap.append(booster)
            booster=[]
        elif "leaf" in line:
            arr=line.replace("\t","").split(":")[0]
            booster.append(arr)
    treeMap.append(booster)
    # 对每棵树中的叶子节点编号、叶子节点index做1个映射，得到映射表。所有的映射表存储在数组treeDic中：
    #       [3,4,5,6] -> { 3:1, 4:2, 5:3, 6:4 }
    #       [7,8,9,10,15,16,12,13,14] -> { 7:1, 8:2, 9:3, 10:4, 15:5, 16:6, 12:7, 13:8, 14:9 }
    #       ...
    treeDic=[]
    for arr in treeMap:
        dic={}
        for i in range(len(arr)):
            dic[arr[i]]=i+1
        treeDic.append(dic)

    return treeDic

if __name__=="__main__":
    tm = parseXgbTreeLeaf("../modelFile/xgb_base_model.txt")
    print tm
