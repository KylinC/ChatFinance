import warnings  # noqa: E501
warnings.filterwarnings('ignore')  # noqa: E501

from jina import DocumentArray, Executor, requests, Flow
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Union, Optional
from torch.nn import Module

import logging
import torch
import pickle
import json
import os


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2, lora_path: Optional[str] = None,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        lora_path = kwargs.get('lora_path', '')
        if not lora_path:
            from accelerate import dispatch_model
            model = AutoModel.from_pretrained(
                checkpoint_path, trust_remote_code=True, **kwargs).half()
            if device_map is None:
                device_map = auto_configure_device_map(num_gpus)
            model = dispatch_model(model, device_map=device_map)

        else:
            from peft import PeftModel
            if device_map is None:
                device_map = auto_configure_device_map(num_gpus)
            model = AutoModel.from_pretrained(
                checkpoint_path, trust_remote_code=True, device_map=device_map).half()
            model = PeftModel.from_pretrained(model, lora_path)

    logging.warn(f"Using Lora From : {lora_path}")

    return model


class ChatGLM2(Executor):
    def __init__(
            self,
            model_name: str = '',
            lora_path: str = '',
            device: str = None,
            num_gpus: int = 0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.shadowmotion = {}
        with open('shadowmotion.pickle', 'rb') as f:
            self.shadowmotion["pickle"] = pickle.load(f)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if device == "cuda":
            self.model = load_model_on_gpus(
                model_name, num_gpus=num_gpus, lora_path=lora_path)
        else:
            self.model = AutoModel(model_name)
            self.model.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

    @requests
    def chat(self, docs: DocumentArray, pre_history: bool = False, **kwargs):
        for doc in docs:
            prompt = doc.text
            history = doc.tags.get('history', [])
            max_length = doc.tags.get('max_length', 8192)
            top_p = doc.tags.get('top_p', 0.95)
            temperature = doc.tags.get('temperature', 0.01)
            if history:
                history = json.loads(doc.tags['history'])
            else:
                if pre_history:
                    pass
                else:
                    # history.append(self.shadowmotion["pickle"])
                    pass

            print('---------prompt----------')

            print(prompt)

            response, history = self.model.chat(
                self.tokenizer, prompt, history=history, max_length=max_length, top_p=top_p, temperature=temperature)

            doc.text = response
            doc.tags['history'] = json.dumps(history, ensure_ascii=False)

            print('--------response---------')

            print(response)

            print('----------end------------')


if __name__ == "__main__":
    model_name = "/home/cql/workspace/others/models/chatglm2-6b"
    # model_name = 'D:\\code\\llm\\chatglm\\chatglm2-6b'
    # lora_path = 'lora'
    lora_path = ''
    port = 50002

    f = Flow(port=port).add(
        uses=ChatGLM2,
        uses_with={
            'model_name': model_name,
            'lora_path': lora_path,
            'device': 'cuda',
            'num_gpus': 1,
        },
        gpus='device=0'
    )

    with f:
        # start server, backend server forever
        f.block()