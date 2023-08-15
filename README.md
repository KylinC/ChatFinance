<p align="center">
  <h1 align="center">ChatFinance</h3>
  <p align="center">金融财报问答大模型</p>
  <p align="center">
  </p>
  <p align="center">
    <a href="https://github.com/KylinC/ChatFinance"><img src="https://img.shields.io/badge/release-v0.0.1-blue" alt="GitHub version"></a>
    <a href="https://github.com/KylinC/ChatFinance"><img src="https://img.shields.io/badge/ROCm-v5.5-orange" alt="GitHub version"></a>
    <a href="https://github.com/KylinC/ChatFinance"><img src="https://img.shields.io/badge/elasticsearch-v8.9.0-yellowgreen" alt="GitHub version"></a>
    <a href="https://github.com/KylinC/eMarkDown"><img src="https://img.shields.io/badge/weaviate-v1.19.5-lightgrey" alt="GitHub version"></a>
  </p>
</p>


### 框架 Framework 



<div align="center"><img src="http://kylinhub.oss-cn-shanghai.aliyuncs.com/uPic/CF.png" style="width=40%;"/></div>

### 模型 Model
使用[ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)模型进行微调，也可以使用base_model进行测试。

```
bash downloads/download_model.sh
```

### 数据 Dataset

原始数据为[ChatGLM评估挑战赛-金融赛道数据集](https://modelscope.cn/datasets/modelscope/chatglm_llm_fintech_raw_dataset/summary)，经过OCR处理之后转化为TXT数据
保留接口：可使用公开金融数据对ElasticSearch和Weaviate数据库进行补充

```
bash downloads/download_data.sh
```


### 推理 Inference


```
vim configs/server.json # 修改base_path和base_python
sudo bash inference_6b.sh 
```

### 使用效果 Demo

```
# 文字概括
{"question": "根据2020年水发燃气的年报，请简要介绍报告期内公司行业格局和趋势的情况。", "answer": "根据2020年水发燃气的年报，公司在行业内的行业格局和趋势情况如下：\n\n1. 行业格局：在能源行业中，水发煤气在规模和市场份额方面占据领先地位。2020年，水发燃气的总资产达到1867亿元人民币，是行业内唯一一家总资产超过1000亿元人民币的公司。同时，水发煤气在煤炭、石油和天然气等领域的市场份额也位居行业前列。\n\n2. 趋势情况：随着能源转型和环保要求的提高，清洁能源已成为能源行业的发展趋势。水发煤气在年报中提到，公司正加大在清洁能源领域的投入，以适应这一趋势。此外，水发煤气还表示，将继续深化在煤炭、石油和天然气等传统领域的业务，以保持公司在能源行业的主导地位。\n\n综上所述，水发煤气在2020年的行业格局中占据了领先地位，同时也在积极应对能源转型和环保要求，加大在清洁能源领域的投入，以适应清洁能源的发展趋势。"}

# 检索
{"question": "江化微2019年研发费用和财务费用分别是多少元?", "answer": "江化微2019年研发费用为5.49亿元，财务费用为1.99亿元。"}

# 开放问题
{"question": "什么是净利润？", "answer": "净利润是指企业在扣除所有成本、税金和其他费用后，所剩余的利润。净利润通常用于衡量企业的盈利能力。"}
```





