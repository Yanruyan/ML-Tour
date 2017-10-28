#!/usr/bin/env bash

# 用libFM训练FM模型（SGD）
../tools/libfm -task c -train ../data/train_data.libfm -test ../data/test_data.libfm -dim '1,1,8'
-method sgd -learn_rate 0.01 -regular 0 -init_stdev 0.001 -iter 1000 -out ../train/out.libfm -rlog ../train/log.txt
