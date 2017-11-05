#!/usr/bin/env bash

# 用libFM训练FM模型（SGD）
# 在公司机器上训练
../tools/libfm/bin/libFM -task r -train train_data.libfm -test test_data.libfm -dim '1,1,8' -method sgd -learn_rate 0.01
-regular 0 -init_stdev 0.001 -iter 100 -save_model fm.model -out result.txt -rlog log.txt