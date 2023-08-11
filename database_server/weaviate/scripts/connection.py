# import sys  # noqa: E501
# sys.path.append('/home/shadowmotion/Documents/code/demo/HRSSC')  # noqa: E501

from langchain.vectorstores import Weaviate
from utils import JinaEmbeddings
from jina import Document
import weaviate

def read_qa_file(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    qa_list = []
    question, answer = None, None
    for line in lines:
        line = line.strip()  # remove leading/trailing whitespaces
        if line.startswith("问："):
            # save the previous qa pair if it exists
            if question and answer:
                qa_list.append(f"{question} {answer}")
            # start a new qa pair
            question = line
            answer = None
        elif line.startswith("答："):
            answer = line
    # don't forget the last qa pair
    if question and answer:
        qa_list.append(f"{question} {answer}")

    return qa_list

client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="shadowmotion-secret-key"))

embedding = JinaEmbeddings("127.0.0.1")
db = Weaviate(client=client, embedding=embedding,
              index_name="LangChain", text_key="text", by_text=False)


# print(embedding.embed_documents(read_qa_file("raw/QA.txt")))

db.add_texts(texts=read_qa_file("./QA.txt"))

# db.add_documents(
#     [Document(page_content="1", metadata={"Q": "1+1=", "A": "2"})]
# )
