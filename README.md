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
bash inference_6b.sh
```







