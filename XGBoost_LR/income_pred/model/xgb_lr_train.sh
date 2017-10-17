#!/bin/bash

# train
../tools/libLinear/liblinear_train -s 6 -c 0.01 -w1 20 ./A2_all_feature_svm.txt ./xgb_lr.model

# predict
../tools/libLinear/liblinear_predict -b 1 ./A3_all_feature_svm.txt ./xgb_lr.model ./A3_predict
cp ./A3_all_feature_svm.txt ./lr_train_tmp

# calculate auc
cat ./A3_predict | sed '1d' | awk '{print $2}' | paste - ./lr_train_tmp/A3_all_feature_svm.txt | awk '{print $2"\t"$1;}' > ./lr_train_tmp/lr.result_compare
cat ./lr_train_tmp/lr.result_compare | ../tools/libLinear/auc


