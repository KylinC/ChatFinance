from models_server.chatglm2.jina_client import encode
from prompts.intent_recognition import intent_recognition_prompt
from prompts.entity_recognition import entity_recognition_prompt
from prompts.answer_generation import answer_generation_prompt
from prompts.open_question import open_question_prompt
from models_server.text2vec.jina_embedding import JinaEmbeddings

from database_server.weaviate.db import insert_table_uuid

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

        docs = search_resp["hits"]["hits"][:3]

        for i, e in enumerate(docs):
            property_name = e['_source']['text']
            company = crawl_name_dict[file_name]
            year = file_name.split("__")[4]+"报"
            property_value = crawl_dict[company][year][property_name]
            # if not property_value or property_value in ["None", "null"]:
            #     continue
            log_file.write(
                f"ES: = = = = = = = = = = = k[{i}] = = = = = = = = = = =\n")
            log_file.write(e['_source']['text'])
            log_file.write("\n")
            extra_information_list.append(f"{property_name}是{property_value}")
    except:
        log_file.write("数据库暂未录入\n")
        
        
    ##################################### -> Embedding 尝试注入
    if not extra_information_list:
    # if True:
        log_file.write("= = EmbeddingInsert = = \n")
        Embedding_Match = False
        if entities[1][-1]=="年":
            target_year = entities[1][:-1]
        target_name = entities[0]
        
        log_file.write(f"尝试搜索{target_year}*{target_name}*.cal\n")
        
        try: 
            target_dir = "/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/alltable"
            pattern = rf'^{target_year}.*{target_name}.*\.cal$'
            
            pattern = os.path.join(target_dir, f"{target_year}*{target_name}*.cal")
            matched_files = [os.path.abspath(path) for path in glob.glob(pattern)]
            insert_table_uuid(matched_files[0],uuid)
            
            log_file.write(f"搜索注入成功,匹配文件名字：{matched_files[0]}\n")
            Embedding_Match = True
        except:
            log_file.write("搜索不到相关.cal文件\n")

    ##################################### -> Embedding Database
    if not extra_information_list and Embedding_Match:
    # if Embedding_Match:
        log_file.write("= = EmbeddingDatabase = = \n")
        index_name = f"LangChain_{uuid}"
        try:
            db = Weaviate(client=client, embedding=embedding,
                          index_name=index_name, text_key="text", by_text=False)

            for word in entities:
                replaced_question = question.replace(word, '')

            docs = db.similarity_search(replaced_question, k=5)

            for i, e in enumerate(docs):
                log_file.write(
                    f"ED: = = = = = = = = = k[{i}] = = = = = = = = =\n")
                log_file.write(e.page_content)
                log_file.write("\n")
                extra_information_list.append(e.page_content)
        except:
            log_file.write("数据库暂未录入\n")

        
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
embedding = JinaEmbeddings("127.0.0.1")
client = weaviate.Client(
    url="http://localhost:50003",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="shadowmotion-secret-key"))

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

bad_ids = [0, 1, 4, 5, 10, 11, 13, 17, 21, 25, 29, 32, 37, 41, 51, 59, 61, 64, 67, 69, 71, 102, 106, 108, 115, 127, 133, 135, 141, 146, 148, 150, 152, 160, 161, 168, 170, 174, 177, 180, 183, 184, 186, 188, 194, 195, 196, 198, 210, 214, 215, 219, 222, 228, 237, 239, 240, 245, 252, 257, 259, 260, 267, 270, 271, 273, 276, 277, 278, 280, 281, 289, 295, 303, 305, 315, 332, 343, 346, 347, 361, 362, 367, 368, 370, 379, 382, 383, 393, 396, 405, 409, 416, 417, 419, 428, 429, 434, 435, 436, 438, 439, 444, 447, 448, 451, 454, 465, 470, 474, 483, 490, 495, 515, 520, 526, 530, 531, 538, 540, 541, 551, 554, 555, 556, 567, 573, 576, 581, 583, 586, 587, 590, 594, 596, 618, 619, 621, 626, 632, 634, 641, 642, 648, 653, 654, 656, 663, 667, 668, 675, 676, 683, 692, 705, 708, 714, 719, 723, 724, 726, 727, 729, 732, 733, 753, 754, 773, 776, 780, 781, 785, 793, 797, 798, 799, 801, 802, 804, 806, 811, 812, 814, 819, 822, 847, 849, 854, 856, 860, 865, 868, 870, 880, 887, 905, 906, 914, 915, 919, 924, 935, 946, 948, 951, 953, 957, 961, 970, 984, 987, 988, 989, 990, 995, 998, 1009, 1011, 1014, 1016, 1022, 1023, 1027, 1032, 1039, 1041, 1043, 1045, 1047, 1048, 1049, 1051, 1054, 1055, 1058, 1060, 1062, 1066, 1067, 1068, 1069, 1072, 1073, 1074, 1078, 1083, 1084, 1088, 1090, 1091, 1093, 1095, 1099, 1102, 1103, 1104, 1121, 1128, 1130, 1131, 1135, 1144, 1146, 1158, 1161, 1162, 1167, 1169, 1171, 1175, 1176, 1178, 1181, 1182, 1186, 1187, 1190, 1193, 1194, 1198, 1199, 1200, 1201, 1203, 1205, 1207, 1208, 1211, 1221, 1227, 1228, 1230, 1232, 1234, 1238, 1242, 1243, 1245, 1247, 1248, 1253, 1258, 1259, 1260, 1261, 1267, 1268, 1269, 1270, 1271, 1277, 1279, 1285, 1290, 1291, 1295, 1296, 1299, 1301, 1302, 1308, 1310, 1312, 1315, 1316, 1320, 1321, 1322, 1323, 1324, 1326, 1328, 1329, 1330, 1332, 1333, 1334, 1338, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1350, 1356, 1357, 1362, 1364, 1365, 1372, 1374, 1377, 1383, 1384, 1385, 1393, 1395, 1400, 1407, 1410, 1412, 1413, 1421, 1423, 1426, 1428, 1438, 1439, 1440, 1442, 1444, 1446, 1453, 1457, 1458, 1459, 1460, 1466, 1474, 1479, 1480, 1481, 1492, 1493, 1495, 1496, 1504, 1505, 1507, 1508, 1510, 1514, 1519, 1522, 1531, 1536, 1540, 1543, 1545, 1549, 1550, 1556, 1558, 1559, 1563, 1564, 1565, 1570, 1574, 1576, 1577, 1582, 1587, 1588, 1594, 1595, 1598, 1599, 1603, 1604, 1606, 1608, 1613, 1614, 1615, 1616, 1624, 1629, 1630, 1633, 1637, 1647, 1651, 1660, 1662, 1665, 1670, 1671, 1673, 1678, 1680, 1681, 1683, 1686, 1693, 1696, 1698, 1701, 1702, 1705, 1708, 1710, 1711, 1716, 1720, 1722, 1728, 1732, 1741, 1742, 1744, 1751, 1754, 1757, 1758, 1760, 1762, 1764, 1767, 1771, 1774, 1777, 1781, 1783, 1790, 1791, 1794, 1797, 1800, 1804, 1805, 1808, 1809, 1810, 1811, 1817, 1820, 1825, 1826, 1827, 1830, 1831, 1833, 1837, 1846, 1850, 1852, 1856, 1858, 1864, 1868, 1872, 1874, 1875, 1876, 1881, 1883, 1885, 1889, 1892, 1893, 1896, 1897, 1901, 1910, 1911, 1914, 1919, 1920, 1926, 1929, 1932, 1938, 1940, 1942, 1943, 1945, 1946, 1952, 1958, 1961, 1962, 1963, 1964, 1965, 1967, 1968, 1971, 1983, 1989, 1996, 1997, 1999, 2002, 2003, 2006, 2014, 2015, 2016, 2025, 2027, 2029, 2031, 2035, 2048, 2062, 2065, 2069, 2071, 2074, 2082, 2086, 2089, 2090, 2092, 2093, 2094, 2096, 2098, 2099, 2105, 2108, 2109, 2111, 2117, 2118, 2119, 2126, 2131, 2132, 2135, 2137, 2142, 2152, 2167, 2182, 2184, 2190, 2199, 2204, 2213, 2214, 2217, 2219, 2221, 2231, 2233, 2234, 2242, 2243, 2244, 2247, 2259, 2268, 2271, 2272, 2282, 2290, 2292, 2294, 2295, 2296, 2297, 2309, 2311, 2312, 2319, 2322, 2324, 2326, 2329, 2333, 2336, 2339, 2340, 2341, 2345, 2346, 2350, 2355, 2367, 2372, 2375, 2379, 2382, 2383, 2386, 2387, 2389, 2402, 2405, 2410, 2413, 2418, 2423, 2425, 2432, 2438, 2440, 2444, 2451, 2452, 2457, 2459, 2463, 2464, 2465, 2467, 2469, 2478, 2480, 2487, 2490, 2502, 2507, 2508, 2509, 2510, 2511, 2517, 2518, 2523, 2530, 2534, 2538, 2539, 2541, 2546, 2548, 2549, 2556, 2559, 2564, 2567, 2570, 2572, 2573, 2575, 2578, 2584, 2586, 2587, 2591, 2598, 2600, 2603, 2611, 2619, 2624, 2629, 2630, 2636, 2640, 2641, 2643, 2644, 2646, 2648, 2655, 2663, 2668, 2671, 2672,
               2674, 2677, 2678, 2679, 2680, 2685, 2686, 2687, 2696, 2701, 2708, 2709, 2712, 2713, 2717, 2720, 2725, 2728, 2729, 2732, 2741, 2742, 2743, 2749, 2757, 2761, 2764, 2771, 2774, 2777, 2781, 2782, 2788, 2790, 2791, 2792, 2795, 2796, 2797, 2801, 2803, 2806, 2807, 2810, 2811, 2812, 2816, 2818, 2821, 2835, 2837, 2838, 2844, 2850, 2852, 2855, 2861, 2867, 2877, 2885, 2886, 2890, 2895, 2902, 2904, 2905, 2906, 2908, 2912, 2917, 2919, 2922, 2923, 2924, 2926, 2927, 2928, 2932, 2933, 2946, 2947, 2950, 2951, 2955, 2957, 2959, 2961, 2967, 2968, 2969, 2975, 2978, 2982, 2986, 2991, 2992, 2994, 2996, 2997, 2998, 3006, 3010, 3012, 3013, 3017, 3018, 3019, 3023, 3026, 3029, 3030, 3031, 3036, 3038, 3040, 3043, 3044, 3050, 3051, 3054, 3056, 3062, 3065, 3068, 3071, 3078, 3079, 3080, 3083, 3085, 3086, 3090, 3111, 3112, 3117, 3118, 3119, 3125, 3127, 3128, 3133, 3135, 3137, 3139, 3150, 3153, 3154, 3156, 3158, 3161, 3164, 3166, 3169, 3174, 3177, 3182, 3188, 3190, 3192, 3195, 3199, 3203, 3205, 3208, 3209, 3211, 3212, 3213, 3215, 3216, 3218, 3225, 3226, 3230, 3231, 3237, 3240, 3243, 3244, 3247, 3248, 3252, 3262, 3268, 3273, 3276, 3277, 3281, 3282, 3285, 3286, 3291, 3292, 3293, 3295, 3296, 3298, 3306, 3310, 3314, 3315, 3316, 3318, 3320, 3321, 3323, 3325, 3334, 3340, 3341, 3342, 3343, 3345, 3352, 3353, 3360, 3361, 3362, 3364, 3366, 3370, 3371, 3373, 3376, 3377, 3383, 3384, 3387, 3388, 3392, 3401, 3404, 3411, 3415, 3418, 3419, 3421, 3424, 3427, 3429, 3436, 3437, 3439, 3440, 3445, 3451, 3460, 3461, 3463, 3467, 3480, 3481, 3482, 3493, 3496, 3498, 3500, 3501, 3502, 3504, 3506, 3512, 3513, 3514, 3517, 3518, 3520, 3521, 3522, 3524, 3527, 3537, 3538, 3541, 3547, 3568, 3569, 3572, 3575, 3576, 3579, 3583, 3585, 3588, 3590, 3591, 3594, 3596, 3605, 3622, 3626, 3632, 3633, 3636, 3643, 3644, 3645, 3648, 3649, 3650, 3653, 3656, 3660, 3661, 3663, 3676, 3687, 3695, 3697, 3703, 3705, 3722, 3724, 3730, 3733, 3734, 3736, 3743, 3745, 3748, 3750, 3758, 3759, 3766, 3773, 3791, 3793, 3798, 3799, 3809, 3812, 3813, 3815, 3817, 3819, 3821, 3824, 3829, 3832, 3833, 3837, 3838, 3842, 3847, 3848, 3851, 3852, 3862, 3865, 3870, 3872, 3873, 3875, 3877, 3880, 3881, 3894, 3896, 3899, 3906, 3910, 3913, 3917, 3920, 3923, 3925, 3941, 3944, 3949, 3951, 3969, 3970, 3975, 3976, 3978, 3982, 3986, 3991, 3992, 3997, 3998, 4002, 4012, 4015, 4019, 4020, 4021, 4023, 4024, 4025, 4034, 4035, 4037, 4038, 4039, 4041, 4045, 4049, 4057, 4062, 4063, 4070, 4071, 4074, 4077, 4079, 4080, 4083, 4085, 4086, 4090, 4095, 4100, 4101, 4103, 4106, 4110, 4115, 4121, 4126, 4140, 4143, 4149, 4153, 4158, 4159, 4161, 4167, 4168, 4170, 4173, 4180, 4184, 4191, 4198, 4199, 4204, 4206, 4211, 4213, 4214, 4217, 4221, 4223, 4224, 4226, 4230, 4231, 4232, 4241, 4242, 4244, 4245, 4247, 4248, 4250, 4254, 4259, 4261, 4262, 4263, 4266, 4267, 4271, 4272, 4279, 4286, 4287, 4292, 4299, 4300, 4304, 4305, 4307, 4308, 4310, 4312, 4313, 4314, 4320, 4328, 4332, 4335, 4340, 4344, 4348, 4349, 4351, 4353, 4362, 4364, 4366, 4370, 4372, 4375, 4376, 4379, 4381, 4382, 4384, 4386, 4399, 4400, 4401, 4404, 4408, 4411, 4412, 4413, 4415, 4418, 4419, 4421, 4422, 4434, 4435, 4437, 4439, 4443, 4446, 4447, 4448, 4455, 4456, 4457, 4462, 4463, 4467, 4468, 4471, 4473, 4474, 4477, 4480, 4482, 4485, 4487, 4495, 4497, 4498, 4499, 4503, 4514, 4525, 4526, 4528, 4529, 4532, 4540, 4545, 4548, 4560, 4563, 4565, 4567, 4569, 4571, 4575, 4583, 4584, 4592, 4593, 4596, 4599, 4600, 4601, 4604, 4609, 4616, 4617, 4619, 4625, 4627, 4630, 4636, 4642, 4647, 4651, 4653, 4654, 4657, 4659, 4667, 4672, 4683, 4685, 4686, 4697, 4699, 4700, 4702, 4711, 4714, 4718, 4727, 4729, 4735, 4738, 4739, 4741, 4748, 4751, 4752, 4753, 4763, 4767, 4769, 4776, 4781, 4784, 4788, 4793, 4796, 4797, 4798, 4800, 4808, 4809, 4812, 4816, 4818, 4822, 4826, 4827, 4831, 4832, 4833, 4836, 4837, 4845, 4846, 4847, 4849, 4852, 4855, 4859, 4860, 4861, 4867, 4868, 4869, 4871, 4875, 4876, 4877, 4884, 4891, 4895, 4896, 4907, 4909, 4913, 4918, 4919, 4922, 4926, 4927, 4934, 4935, 4936, 4944, 4945, 4946, 4957, 4959, 4962, 4964, 4965, 4966, 4971, 4973, 4974, 4975, 4985, 4986, 4995, 4999]


with open(f"./logs/log_{formatted_time}.txt", "w") as log_file, open(f"./logs/submission_{formatted_time}.json", "w") as sm_file, open("./data/chatglm_llm_fintech_raw_dataset/test_questions.jsonl", "r") as qs_file:
    question_count = 0
    for question_line in qs_file:
        question_count += 1
        ##### id 截断
        # if question_count<1734:
        #     continue
        print("question_count:",question_count)
        question_dict = json.loads(question_line)
        ##### bad id 截断
        if question_dict["id"] not in bad_ids:
            continue
        answer = generate(question_dict["question"], uuid_dict, crawl_dict, crawl_name_dict, es, log_file)
        answer_dict = {"id":question_dict["id"],"question":question_dict["question"],"answer":answer}
        sm_file.write(f"{answer_dict}\n")
        time.sleep(3)


