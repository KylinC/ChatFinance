import sys  # noqa: E501
# sys.path.append('/home/shadowmotion/Documents/code/demo/HRSSC')  # noqa: E501


from langchain.vectorstores import Weaviate
from langchain.schema import Document
from utils import JinaEmbeddings
import weaviate
import json
import os

client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="kylin-secret-key"))

embedding = JinaEmbeddings("127.0.0.1")

with open("../../data/chatglm_llm_fintech_raw_dataset/uuid.json", "r", encoding='utf-8') as f:
    uuid_dict = json.load(f)

query_list = [
    "公司的法定代表人是谁",
    "电子邮箱是什么",
    "公司的外文名称是什么",
]


index_name = "LangChain_135087231333628284559671447376917039719"

db = Weaviate(client=client, embedding=embedding,
              index_name=index_name, text_key="text", by_text=False)

for query in query_list[:1]:

    docs = db.similarity_search(query, k=3)

    print(f" >>>>>>>>>>> {query} <<<<<<<<<<<<")

    for i, e in enumerate(docs):
        print(f" = = = = = = = = = = = k[{i}] = = = = = = = = = = =")
        print(e.page_content)
