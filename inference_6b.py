from models_server.chatglm2.jina_client import encode
from prompts.intent_recognition import intent_recognition_prompt
from prompts.entity_recognition import entity_recognition_prompt
from prompts.answer_generation import answer_generation_prompt
from prompts.open_question import open_question_prompt
from models_server.text2vec.jina_embedding import JinaEmbeddings

from database_server.weaviate.db import insert_table

from langchain.vectorstores import Weaviate
from elasticsearch import Elasticsearch

import weaviate
import json
import os
import glob


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


def generate(question, uuid_dict, crawl_dict, crawl_name_dict, es, log_file):
    log_file.write("= = 流程开始 = = \n")
    log_file.write(f"Q:\n{question}\n\n")

    # -> Intent Recognition
    log_file.write("= = 意图识别 = = \n")
    prompt = intent_recognition_prompt(question)
    response = encode(prompt, history=[])
    log_file.write(f"R:\n{response[0].text}\n\n")

    if "检索问题" not in parse_intent_recognition(response[0].text):
        log_file.write("开放问题直接作答\n")
        prompt = open_question_prompt(question)
        response = encode(prompt, history=[])
        answer = response[0].text
        log_file.write(f"R:\n{answer}\n\n")
        return answer
    
    # print("意图识别时间：",time.time()-initial_time)

    ############################ -> Entity Recognition
    log_file.write("= = 实体提取 = = \n")
    prompt = entity_recognition_prompt(question)
    response = encode(prompt, history=[])
    log_file.write(f"R:\n{response[0].text}\n\n")
    entities = parse_entity_recognition(response[0].text)
    uuid, file_name = attain_uuid(entities, uuid_dict)
    log_file.write(f"R:\n{uuid}\n\n")
    if not uuid and entities[0][0] == '年':
        entities[0] = entities[0][1:]
        uuid, file_name = attain_uuid(entities, uuid_dict)
        log_file.write(f"R:\n 修复公司名称： {entities[0]}\n\n")
    if not uuid:
        log_file.write("未知公司不予作答\n")
        return ""
    
    # print("实体提取时间：",time.time()-initial_time)

    elastic_search_success = False
    extra_information_list = []

    ################################ -> ElasticSearch
    log_file.write("= = ElasticSearch = = \n")
    index_name = f"{uuid}"
    # index_name = "all_property"
    try:
        for word in entities:
            replaced_question = question.replace(word, '')

        search_query = {
            "query": {
                "match": {
                    "text": replaced_question
                }
            }
        }

        search_resp = es.search(index=index_name, body=search_query)

        docs = search_resp["hits"]["hits"][:5]

        for i, e in enumerate(docs):
            property_name = e['_source']['text']
            company = crawl_name_dict[file_name]
            year = file_name.split("__")[4]+"报"
            property_value = crawl_dict[company][year][property_name]
            if not property_value or property_value in ["None", "null"]:
                    continue
            log_file.write(
                f"ES: = = = = = = = = = = = k[{i}] = = = = = = = = = = =\n")
            log_file.write(e['_source']['text'])
            log_file.write("\n")
            extra_information_list.append(f"{property_name}是{property_value}")
    except:
        log_file.write("数据库暂未录入\n")
        
        
    ##################################### -> Embedding 尝试注入
    
    if entities[1][-1]=="年":
        target_year = entities[1][:-1]
    target_name = entities[0]
    
    log_file.write(f"尝试搜索{target_year}*{target_name}*.cal\n")
    
    try: 
        target_dir = "/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/alltable"
        pattern = rf'^{target_year}.*{target_name}.*\.cal$'
        
        pattern = os.path.join(target_dir, f"{target_year}*{target_name}*.cal")
        matched_files = [os.path.abspath(path) for path in glob.glob(pattern)]
        insert_table(matched_files[0],uuid_dict)
        
        log_file.write(f"搜索注入成功,匹配文件名字：{matched_files[0]}\n")
    except:
        log_file.write("搜索不到相关.cal文件\n")

    ##################################### -> Embedding Database
    # f.write("= = EmbeddingDatabase = = \n")
    # if not elastic_search_success and not extra_information_list:
    #     index_name = f"LangChain_{uuid}"
    #     try:
    #         db = Weaviate(client=client, embedding=embedding,
    #                       index_name=index_name, text_key="text", by_text=False)

    #         for word in entities:
    #             replaced_question = question.replace(word, '')

    #         docs = db.similarity_search(replaced_question, k=5)

    #         for i, e in enumerate(docs):
    #             f.write(
    #                 f"ED: = = = = = = = = = = = k[{i}] = = = = = = = = = = =\n")
    #             f.write(e.page_content)
    #             f.write("\n")
    #             extra_information_list.append(e.page_content)
    #     except:
    #         f.write("数据库暂未录入\n")

    #     response = encode(question, history=[])
    #     answer = response[0].text
        
    # print("向量库搜索时间：",time.time()-initial_time)

    log_file.write("= = AnswerGeneration = = \n")
    extra_information = "\n".join(extra_information_list)
    log_file.write(extra_information+'\n')
    prompt = answer_generation_prompt(extra_information, question)
    response = encode(prompt, history=[])
    log_file.write(f"R:\n{response[0].text}\n\n")
    answer=response[0].text
    return answer


# import time
# initial_time = time.time()

# -> Init Embedding Database
# embedding = JinaEmbeddings("127.0.0.1")
# client = weaviate.Client(
#     url="http://localhost:50003",  # Replace with your endpoint
#     auth_client_secret=weaviate.AuthApiKey(api_key="shadowmotion-secret-key"))

# print("向量库时间：",time.time()-initial_time)

# -> Init Embedding Database
es = Elasticsearch('http://localhost:50004')

# print("es时间：",time.time()-initial_time)

# -> Init UUID Dict
with open("./data/chatglm_llm_fintech_raw_dataset/uuid.json", "r") as f:
    uuid_dict = json.load(f)

# -> Init crawl Dict
with open("./data/chatglm_llm_fintech_raw_dataset/allcrawl.json", "r") as f:
    crawl_dict = json.load(f)
with open("./data/chatglm_llm_fintech_raw_dataset/name_map_crawl.json", "r") as f:
    crawl_name_dict = json.load(f)
    
# print("dict时间：",time.time()-initial_time)

# question = "本钢板材在2020年对联营企业和合营企业的投资收益是多少元？"

import time
from datetime import datetime
formatted_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

with open(f"./logs/log_{formatted_time}.txt", "w") as log_file, open(f"./logs/submission_{formatted_time}.json", "w") as sm_file, open("./data/chatglm_llm_fintech_raw_dataset/test_questions.jsonl", "r") as qs_file:
    question_count = 0
    for question_line in qs_file:
        question_count += 1
        if question_count<1734:
            continue
        print("question_count:",question_count)
        question_dict = json.loads(question_line)
        answer = generate(question_dict["question"], uuid_dict, crawl_dict, crawl_name_dict, es, log_file)
        answer_dict = {"id":question_dict["id"],"question":question_dict["question"],"answer":answer}
        sm_file.write(f"{answer_dict}\n")
        time.sleep(3)


