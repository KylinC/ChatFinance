#!/bin/bash

cd ../models || mkdir ../models

git clone https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase

echo "Embedding Model(for vector DB) downloaded!"

git clone https://huggingface.co/THUDM/chatglm2-6b

echo "ChatGML-6B Model(for vector DB) downloaded!"