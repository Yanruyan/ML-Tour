#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 提取FM组合的效果好的组合特征

import util.parseFm

if __name__ == "__main__":
    # 解析FM模型
    modelPath = "../data/cvr_fm.model"
    (w0,w,cw) = util.parseFm.parseFmModel(modelPath)
    # print(w0)
    # print(w)
    # f = open("../data/cw.txt",'w')
    # for vec in cw:
    #     str1 = ""
    #     for elem in vec:
    #         str1 = str1 + " " + str(elem)
    #     f.write(str1)
    #     f.write("\n")
    # 计算交叉特征权重
    # crossWeights = util.parseFm.calCrossWeight(cw)
    # f2 = open("../data/cross_weight.txt",'w')
    # for wi in crossWeights:
    #     str2 = "["
    #     for elem in wi:
    #         str2 = str2 + " " + str(elem)
    #     str2 = str2 + "]"
    #     f2.write(str2)
    #     f2.write("\n")
    # f2.close()
    # 卖家活跃时间特征与其他特征交叉后的权重
    featCount = len(cw)
    rows = list(range(featCount-26,featCount-20))
    cols = list(range(featCount-20,featCount))
    crossWeight2 = util.parseFm.calCrossWeightBySelf(cw,rows,cols)
    f3 = open("../data/cross_weight3.txt",'w')
    for wi in crossWeight2:
        str2 = "["
        for elem in wi:
            str2 = str2 + " " + str(elem)
        str2 = str2 + "]"
        f3.write(str2)
        f3.write("\n")
    f3.close()










