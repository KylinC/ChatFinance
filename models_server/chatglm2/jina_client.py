# -*- coding: utf-8 -*-

from jina import Document, DocumentArray
from jina import Client
import sys
import time

sys.path.append('..')

port = 50002
c = Client(port=port)


def encode(sentence, history):
    """Get one sentence embeddings from jina server."""
    r = c.post(
        '/', inputs=DocumentArray([Document(text=sentence, tags={"history": history})]))
    return r


if __name__ == '__main__':
    # 我们创建一个会话的历史，你可以根据需要更改
    history = []

    # 发起一个聊天会话
    sentences = ['你好', '中国人认为宇宙万法的那个源头，它是什么', '你跟我说说这宇宙万物的本源是什么？']
    for sent in sentences:
        # 创建请求，发送给executor
        response = encode(sent, history)

        # 打印返回的响应
        print(f"Response: {response[0].text}")
        print(f"Updated history: {response[0].tags['history']}")

        # 更新会话历史
        history = response[0].tags['history']
