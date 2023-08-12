import warnings  # noqa: E501
warnings.filterwarnings('ignore')  # noqa: E501

from langchain.embeddings.base import Embeddings
from docarray import Document, DocumentArray
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
        embedding = self._post(docs).embeddings[0]
        return list(map(float, embedding))