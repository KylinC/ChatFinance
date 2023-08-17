import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

import warnings
warnings.filterwarnings("ignore")

dftrain = pd.read_parquet('/home/kylin/workspace/ChatFinance/data/sft/intent_sft_10k.parquet')
dftest = pd.read_parquet('/home/kylin/workspace/ChatFinance/data/sft/intent_sft_10k_val.parquet')

# model.chat
def build_inputs(query, history):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response) # history中的第几轮次，问了什么，得到了什么答案
    prompt += "[Round {}]\n\n问：{} -> \n\n答：".format(len(history) + 1, query) # 当前轮次，当前问话
    return prompt

his = [("文本分类任务：对一段问题进行意图识别，分成开放问题或者检索问题。\n\n下面是一些范例:\n\n什么是投资比率？ -> 开放问题\n快手科技2021年的营业额是多少？  -> 检索问题\n利润率是指什么？ -> 开放问题\n百度集团2021年的硕士生人数比例是多少 -> 检索问题\n\n请对以下问题进行分类。返回'开放问题'或者'检索问题'，无需其它说明和解释。\n\n什么是股东权益？ ->\n\n", 'n什么是股东权益？ -> 开放问题')]
dftrain['context'] = [build_inputs(x,history=his) for x in dftrain['text']] # 定义训练集中的上文
dftrain['target'] = [x for x in dftrain['tag']] # 定义训练集中的标签
dftrain = dftrain[['context','target']]

dftest['context'] = [build_inputs(x,history=his) for x in dftest['text']]
dftest['target'] = [x for x in dftest['tag']]
dftest = dftest[['context','target']]

ds_train = datasets.Dataset.from_pandas(dftrain)
ds_val = datasets.Dataset.from_pandas(dftest)

model_name = '/home/kylin/workspace/ChatFinance/models/chatglm2-6b'
max_seq_length = 512
skip_over_length = True
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')

def preprocess(example):
    context = example["context"]
    target = example["target"]
    context_ids = tokenizer.encode(
            context,
            max_length=max_seq_length,
            truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = context_ids + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids, "context_len": len(context_ids),'target_len':len(target_ids)}

ds_train_token = ds_train.map(preprocess).select_columns(['input_ids', 'context_len','target_len'])
if skip_over_length: 
    ds_train_token = ds_train_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)
    
ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'context_len','target_len'])
if skip_over_length:
    ds_val_token = ds_val_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)

def data_collator(features: list):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for length, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        context_len = feature["context_len"]
        labels = (
            [-100] * context_len + ids[context_len :] + [-100] * (longest - length)
        ) 
        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }
    
# ds_train_token 是送入训练的数据集
# num_workers 是数据载入时将使用多线程并行处理，这可以在一定程度上加速数据载入
# batch_size 是每一个批次的样本数量
# pin_memory=True: 如果设为 True，那么数据载入器将会在返回Tensor之前，先将其复制到CUDA固定内存中。这样可以使得转移数据到GPU上更快
# shuffle=True: 如果设为 True，那么在每个训练周期开始时，数据载入器将会打乱数据集的顺序
# collate_fn=data_collator: 这个函数定义了如何将多个样本合并成一个小批量。在这里，我们使用之前定义的 data_collator 函数，这个函数会按照我们的需要对每个小批量的数据进行预处理
dl_train = torch.utils.data.DataLoader(ds_train_token,num_workers=2,batch_size=4,
                                       pin_memory=True,shuffle=True,
                                       collate_fn = data_collator)
dl_val = torch.utils.data.DataLoader(ds_val_token,num_workers=2,batch_size=4,
                                    pin_memory=True,shuffle=True,
                                     collate_fn = data_collator)

dl_train.size = 300 #用约300个step做一次验证

import locale
locale.getpreferredencoding = lambda: "UTF-8"

model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False,
                                  trust_remote_code=True)

#节约cuda，但可能会使得训练时间变长
model.supports_gradient_checkpointing = True  
model.gradient_checkpointing_enable() 
model.enable_input_require_grads() 

# 关闭了模型的缓存机制，该设置可以避免一些警告，但在模型推理时需要重新开启
model.config.use_cache = False  

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

# 开启模型的并行处理能力，这可以在有多个GPU的情况下提高训练效率
model.is_parallelizable = True
model.model_parallel = True


# model.print_trainable_parameters()
# 可训练参数：1949696
# 总参数量：6245533696
# 需要调整的模型参数量的占比 3.1%

from torchkeras import KerasModel
from accelerate import Accelerator

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None,
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage=='train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"],labels=batch["labels"]).loss

        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        step_losses = {self.stage+"_loss":all_loss.item()}

        step_metrics = {}

        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics

KerasModel.StepRunner = StepRunner


def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path,'adapter_model.bin')),strict =False)
    self.from_scratch = False

KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt


keras_model = KerasModel(model,loss_fn = None,
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6))
ckpt_path = '~/.ckpt/chatglm2_intent10k'


keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=100,patience=5,
                monitor='val_loss',mode='min',
                ckpt_path = ckpt_path,
                mixed_precision='fp16'
               )

model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False,
                                  trust_remote_code=True,
                                  device_map='auto')
model = PeftModel.from_pretrained(model,ckpt_path)
model = model.merge_and_unload() #合并lora权重

model.save_pretrained("../models/sft/chatglm2-6b-intent10k", max_shard_size='1GB')
tokenizer.save_pretrained("../models/sft/chatglm2-6b-intent10k")