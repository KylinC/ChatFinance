import warnings  # noqa: E501
warnings.filterwarnings('ignore')  # noqa: E501

from langchain.embeddings.base import Embeddings
from jina import Document, DocumentArray
from jina import Client

from typing import Any, List


class JinaEmbeddings(Embeddings):
    def __init__(self, host: str = "0.0.0.0", port: int = 50001, **kwargs: Any) -> None:
        self.client = Client(host=host, port=port, **kwargs)

    def _post(self, docs: List[Any], **kwargs: Any) -> Any:
        payload = dict(inputs=docs, **kwargs)
        return self.client.post(on="/", **payload)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        docs = DocumentArray([Document(text=t) for t in texts])
        embeddings = self._post(docs).embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        docs = DocumentArray([Document(text=text)])
        print(docs)
        embedding = self._post(docs).embeddings[0]
        return list(map(float, embedding))


if __name__ == "__main__":
    embedding = JinaEmbeddings("127.0.0.1")

    eg = "嵌入模型（Embedding model）通常用于将词语或者短语转化为向量表示。嵌入模型通常不会有严格的输入长度限制，因为它主要关注的是如何将单个词或短语转化为向量表示。然而，在某些应用中，嵌入模型可能会在更大的上下文环境中考虑单词，这时可能会有输入长度的限制。如果你使用的是一些预训练的模型，如BERT、GPT等，它们在实际训练过程中会有一个最大序列长度限制，这是由于这些模型的结构决定的。例如，BERT模型的最大输入长度通常设定为512个词语。如果提供的输入序列长度超过这个限制，那么可能需要进行截断，或者采用其他处理策略。如果你的输入长度超过了这个限制，直接输入给模型，可能会导致出错，或者导致模型无法处理那些超出长度限制的部分，因此，通常我们在数据预处理阶段就要处理好这个问题，确保所有输入都不超过模型的长度限制。"

    print(len(eg))

    r = embedding.embed_query(eg)

    print(len(r))
