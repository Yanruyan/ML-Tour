#!/bin/bash

# train
../tools/libLinear/liblinear_train -s 6 -c 0.01 -w1 20 ./A2_train_svm_data.txt ./lr_base.model
# predict
../tools/libLinear/liblinear_predict -b 1 ./A2_test_svm_data.txt ./lr_base.model ./A2_test_predict
cp ./A2_test_svm_data.txt ./lr_train_tmp

# calculate auc
cat ./A2_test_predict | sed '1d' | awk '{print $2}' | paste - ./lr_train_tmp/A2_test_svm_data.txt | awk '{print $2"\t"$1;}' > ./lr_train_tmp/lr.result_compare
cat ./lr_train_tmp/lr.result_compare | ../tools/libLinear/auc


