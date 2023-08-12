#!/bin/bash

JSON_FILE="configs/server.json"
LOGS="logs"
BASE_PATH=$(jq -r '.base_path' $JSON_FILE)
PYTHON_PATH=$(jq -r '.base_python' $JSON_FILE)
ELASTIC_SEARCH_PATH=$(jq -r '.sever_path.elastic_search' $JSON_FILE)
WEAVIATE_PATH=$(jq -r '.sever_path.weaviate' $JSON_FILE)
LLM_SERVER=$(jq -r '.sever_path.chatglm2' $JSON_FILE)
TEXT_SERVER=$(jq -r '.sever_path.text2vec' $JSON_FILE)


# 启动 text2vec model (for WEAVIATE)
cd "$BASE_PATH/$TEXT_SERVER" && nohup $PYTHON_PATH jina_server.py > "$BASE_PATH/$LOGS/text2vec.log" 2>&1 &
echo "text2vec model start!"

# 启动 elastic search
cd "$BASE_PATH/$ELASTIC_SEARCH_PATH" && docker-compose up -d
echo "elastic DB start!"

# 启动 chatgml-6b
cd "$BASE_PATH/$LLM_SERVER" && nohup $PYTHON_PATH jina_server.py > "$BASE_PATH/$LOGS/llm.log" 2>&1 &
echo "gml-6b model start!"

# 启动 weaviate
cd "$BASE_PATH/$WEAVIATE_PATH" && docker-compose up -d
echo "weaviate DB start!"
echo "====================================="

# 启动 生成程序
cd "$BASE_PATH"
$PYTHON_PATH inference_6b.py