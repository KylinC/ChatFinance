from models_server.chatglm2.jina_client import encode
from prompts.intent_recognition import intent_recognition_prompt
from prompts.entity_recognition import entity_recognition_prompt
from prompts.answer_generation import answer_generation_prompt
from models_server.text2vec.jina_embedding import JinaEmbeddings

from langchain.vectorstores import Weaviate
from elasticsearch import Elasticsearch

import weaviate
import json

def parse_entity_recognition(response: str):
    parse_list = []
    lines = response.split('\n')
    for line in lines:
        sep = ':' if ':' in lines[-1] else '：'
        if "公司名" in line:
            parse_list.append(line.split(sep)[1])
        if "年份" in line:
            parse_list.append(line.split(sep)[1])
    return parse_list

def parse_intent_recognition(response: str):
    lines = response.split('\n')
    return lines[-1]


def attain_uuid(entities, uuid_dict):
    for k, v in uuid_dict.items():
        fg = True
        for entity in entities:
            if entity not in k:
                fg = False
                break
        if fg:
            print(entities, k)
            return v, k
    return None, None