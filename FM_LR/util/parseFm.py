#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 解析FM模型，获取一阶、二阶特征的权重.

def parseFmModel( modelPath ):
    modelFile = open(modelPath,'r')
    modelLines = modelFile.readlines()
    isW0 = False
    isW = False
    isCW = False
    w0 = -1.0
    w = []
    cw = []
    for line in modelLines:
        if( "global bias" in line ):
            isW0 = True
        elif( "unary interactions Wj" in line ):
            isW = True
            isW0 = False
        elif( "pairwise interactions Vj,f" in line ):
            isCW = True
            isW = False
            isW0 = False
        # 获取bias的权重
        else:
            if( isW0 ):
                w0 = float(line)
            if( isW ):
                w.append( float(line) )
            if( isCW ):
                fields = line.split(" ")
                vec = []
                for elem in fields:
                    vec.append( float(elem) )
                cw.append(vec)
    return (w0,w,cw)

def calCrossWeight( featVectors ):
    count = len(featVectors)
    crossWeights = []
    for i in list(range(0,count-1)):
        wi = []
        vi = featVectors[i]
        for j in list(range(i+1,count)):
            vj = featVectors[j]
            wi.append( vectorInnerProduct(vi,vj) )
        crossWeights.append(wi)
    return (crossWeights)

def calCrossWeightBySelf( featVectors, rows, cols ):
    crossWeights = []
    for i in rows:
        wi = []
        vi = featVectors[i]
        for j in cols:
            vj = featVectors[j]
            wi.append(vectorInnerProduct(vi, vj))
        crossWeights.append(wi)
    return (crossWeights)

def vectorInnerProduct( vec1, vec2 ):
    innProd = 0.0
    count = len(vec1)
    for idx in list(range(0,count)):
        innProd = innProd + vec1[idx]*vec2[idx]
    return (innProd)

