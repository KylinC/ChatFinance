from jina import DocumentArray, Executor, requests
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
from jina import Flow
import numpy as np
import torch
import os
import json


class Text2vecEncoder(Executor):
    """The Text2vecEncoder encodes sentences into embeddings using transformers models."""

    def __init__(
            self,
            model_name,
            base_tokenizer_model: Optional[str] = None,
            pooling_strategy: str = 'mean',
            layer_index: int = -1,
            max_length: Optional[int] = 256,
            embedding_fn_name: str = '__call__',
            device: str = None,
            traversal_paths: str = '@r',
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        """
        The transformer torch encoder encodes sentences into embeddings.

        :param model_name: Name of the pretrained model or path to the
            model
        :param base_tokenizer_model: Base tokenizer model
        :param pooling_strategy: The pooling strategy to be used. The allowed values are
            ``'mean'``, ``'min'``, ``'max'`` and ``'cls'``.
        :param layer_index: Index of the layer which contains the embeddings
        :param max_length: Max length argument for the tokenizer, used for truncation. By
            default the max length supported by the model will be used.
        :param embedding_fn_name: Function to call on the model in order to get output
        :param device: Torch device to put the model on (e.g. 'cpu', 'cuda', 'cuda:1')
        :param traversal_paths: Used in the encode method an define traversal on the
             received `DocumentArray`
        :param batch_size: Defines the batch size for inference on the loaded
            PyTorch model.
        """
        super().__init__(*args, **kwargs)

        self.traversal_paths = traversal_paths
        self.batch_size = batch_size

        base_tokenizer_model = base_tokenizer_model or model_name

        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.embedding_fn_name = embedding_fn_name

        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_model)
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.model.to(device).eval()

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """
        Encode text data into a ndarray of `D` as dimension, and fill the embedding of
        each Document.

        :param docs: DocumentArray containing text
        :param parameters: dictionary to define the `traversal_paths` and the
            `batch_size`. For example,
            `parameters={'traversal_paths': 'r', 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """

        docs_batch_generator = DocumentArray(
            filter(
                lambda x: bool(x.text),
                docs[parameters.get('traversal_paths', self.traversal_paths)],
            )
        ).batch(batch_size=parameters.get('batch_size', self.batch_size))

        for batch in docs_batch_generator:
            texts = batch.texts

            with torch.inference_mode():
                input_tokens = self._generate_input_tokens(texts)
                outputs = getattr(self.model, self.embedding_fn_name)(
                    **input_tokens)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()
                hidden_states = outputs.hidden_states
                embeds = self._compute_embedding(hidden_states, input_tokens)
                batch.embeddings = embeds

    def _compute_embedding(
            self, hidden_states: Tuple['torch.Tensor'], input_tokens: Dict
    ):
        fill_vals = {'cls': 0.0, 'mean': 0.0, 'max': -np.inf, 'min': np.inf}
        fill_val = torch.tensor(
            fill_vals[self.pooling_strategy], device=self.device)
        layer = hidden_states[self.layer_index]

        attn_mask = input_tokens['attention_mask']

        # Fix LongFormerModel like model which has mismatch seq_len between
        # attention_mask and hidden_states
        padding_len = layer.size(1) - attn_mask.size(1)
        if padding_len > 0:
            attn_mask = torch.nn.functional.pad(
                attn_mask, (0, padding_len), value=0)

        expand_attn_mask = attn_mask.unsqueeze(-1).expand_as(layer)

        layer = torch.where(expand_attn_mask.bool(), layer, fill_val)
        embeddings = layer.sum(dim=1) / expand_attn_mask.sum(dim=1)
        return embeddings.cpu().numpy()

    def _generate_input_tokens(self, texts):
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        input_tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )

        input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
        return input_tokens


with open('../../configs/server.json', 'r') as file:
    server_config = json.load(file)
base_path = server_config["base_path"]
model_path = os.path.join(base_path,server_config["models_path"]["text2vec"])
print(model_path)
# port = server_config["port"]["text2vec"]
lora_path = ""

f = Flow(port=50001).add(
    uses=Text2vecEncoder,
    uses_with={
        'model_name': model_path,
        'device': 'cuda',
    },
    gpus='device=0',
)

with f:
    # start server, backend server forever
    f.block()
