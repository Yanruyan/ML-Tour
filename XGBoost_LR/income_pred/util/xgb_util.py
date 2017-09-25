#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 与xgboost相关的一些工具


#######################################################################################################################
# 将xgboost的预测结果（预测样本落在树的哪个叶子上）写入文件
def dumpPredictLeaf2File( predictByLeaf, filePath ):
    rows = len(predictByLeaf[:, 0])
    cols = len(predictByLeaf[0, :])
    row_idx = 0
    ff = open(filePath, 'a')
    while (row_idx < rows):
        col_idx = 0
        row_string = ""
        while (col_idx < cols):
            row_string = row_string + ("%d " % (predictByLeaf[row_idx, col_idx]))
            col_idx = col_idx + 1
        ff.write(row_string)
        ff.write("\n")
        row_idx = row_idx + 1
    ff.close()

#######################################################################################################################
#

