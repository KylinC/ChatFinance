import sys  # noqa: E501
sys.path.append('/home/kylin/workspace/ChatFinance')  # noqa: E501

from langchain.vectorstores import Weaviate
from utils import JinaEmbeddings
from jina import Document
import weaviate
import glob
import json
import os


client = weaviate.Client(
    url="http://localhost:50003",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="shadowmotion-secret-key"))

embedding = JinaEmbeddings("127.0.0.1")


# print(embedding.embed_documents(read_qa_file("raw/QA.txt")))


def insert_txt(path, uuid_dict):

    basename = os.path.basename(path).split('.')[0]

    db = Weaviate(client=client, embedding=embedding,
                  index_name=f"LangChain_{uuid_dict[basename]}", text_key="text", by_text=False)
    print(f"To insert -> {basename}")
    print(f"index_name: {db._index_name}")

    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0 and i % 1000 == 0:
                db.add_texts(texts=texts)
                print(f"文字数据已注入{i}")
                texts = []
            if len(line) <= 1:
                continue
            texts.append(line[:-1])
        db.add_texts(texts=texts)
        print(f"文字数据已注入{i}")
        texts = []


def insert_table(path, uuid_dict):
    basename = os.path.basename(path).split('.')[0]

    db = Weaviate(client=client, embedding=embedding,
                  index_name=f"LangChain_{uuid_dict[basename]}", text_key="text", by_text=False)
    print(f"To insert -> {basename}")
    print(f"index_name: {db._index_name}")

    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0 and i % 1000 == 0:
                db.add_texts(texts=texts)
                print(f"表格数据已注入{i}")
                texts = []
            if len(line) <= 1:
                continue
            texts.append(line[:-1])
        db.add_texts(texts=texts)
        print(f"表格数据已注入{i}")
        texts = []

def insert_table_uuid(path, uuid):
    basename = os.path.basename(path).split('.')[0]

    db = Weaviate(client=client, embedding=embedding,
                  index_name=f"LangChain_{uuid}", text_key="text", by_text=False)
    print(f"To insert -> {basename}")
    print(f"index_name: {db._index_name}")

    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0 and i % 1000 == 0:
                db.add_texts(texts=texts)
                # print(f"表格数据已注入{i}")
                texts = []
            if len(line) <= 1:
                continue
            texts.append(line[:-1])
        db.add_texts(texts=texts)
        print(f"表格数据已注入{i}")
        texts = []


if __name__ == "__main__":
    base_tokenizer_model = '/home/kylin/workspace/ChatFinance/models/text2vec-base-chinese-paraphrase'

    with open("/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/uuid.json", "r", encoding='utf-8') as f:
        uuid_dict = json.load(f)

    n = 30000
    skip = 0

    # TXT_DIRECTORY = "/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/alldata"
    # file_names = glob.glob(TXT_DIRECTORY + '/*')
    # for i, file_name in enumerate(file_names):
    #     print(f"No.{i} insert_txt")
    #     try:
    #         insert_txt(file_name, uuid_dict)
    #     except:
    #         print(f"error: {file_name}")
    #     if i >= n - 1:
    #         break

    TAB_DIRECTORY = "/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/alltable"
    file_names = glob.glob(TAB_DIRECTORY + '/*.cal')
    print(file_names)
    for i, file_name in enumerate(file_names):
        if i < skip:
            continue
        print(f"No.{i} insert_tab")
        try:
            insert_table(file_name, uuid_dict)
        except:
            print(f"error: {file_name}")
        if i >= n - 1:
            break
