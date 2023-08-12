import sys  # noqa: E501
sys.path.append("/home/kylin/workspace/ChatFinance")  # noqa: E501
from elasticsearch import Elasticsearch

import json


def attain_uuid(entities, uuid_dict):
    for k, v in uuid_dict.items():
        fg = True
        for entity in entities:
            if entity not in k:
                fg = False
                break
        if fg:
            print(entities, k)
            return v
    return None


if __name__ == "__main__":
    es = Elasticsearch('http://localhost:50004')

    with open("/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/uuid.json", "r") as f:
        uuid_dict = json.load(f)

    with open("/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/allcrawl.json", "r") as f:
        crawl_dict = json.load(f)

    for i, company in enumerate(crawl_dict):
        for year in crawl_dict[company]:
            if year not in ["2019年报", "2020年报", "2021年报"]:
                continue
            try:
                uuid = attain_uuid(
                    [crawl_dict[company][year]['SECURITY_CODE'], year[:-1]], uuid_dict)
                for idx, key in enumerate(crawl_dict[company][year]):
                    doc = {
                        "text": key,
                    }
                    resp = es.index(index=str(uuid), id=idx, document=doc)
            except:
                print(f"error {company} {year}")
        if i % 99 == 0 and i > 0:
            print(f"insert {3*(i+1)} file")
    print(f"insert {3*len(crawl_dict)} file")
