#!/bin/bash

cd ../data || mkdir ../data

git clone http://www.modelscope.cn/datasets/modelscope/chatglm_llm_fintech_raw_dataset.git

echo "PDF data downloaded!"