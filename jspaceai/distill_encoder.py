"""小模型感知编码器——把小模型表征投影到 input_dim"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class SmallModelEncoder(nn.Module):
    def __init__(self, input_dim=32, model_name="gpt2", use_real_model=True):
        super().__init__()
        self.input_dim = input_dim
        self.model_name = model_name
        self._real_model = None
        self._tokenizer = None
        self._hidden_size = 128

        if use_real_model:
            try:
                from transformers import AutoModel, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
                self._model.eval()
                self._hidden_size = self._model.config.hidden_size
                self._real_model = self._model
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                print(f"  [编码器] 已加载 {model_name} (hidden={self._hidden_size})")
            except Exception as e:
                print(f"  [编码器] 降级: {e}")

        if self._real_model is not None:
            self.proj = nn.Linear(self._hidden_size, input_dim, bias=False)
        else:
            self.embed = nn.Embedding(1000, 64)
            self.proj = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))

    def forward(self, input_data):
        if self._real_model is not None:
            if isinstance(input_data, str):
                input_data = [input_data]
            if isinstance(input_data, list):
                with torch.no_grad():
                    inputs = self._tokenizer(input_data, return_tensors="pt",
                                           truncation=True, max_length=64, padding=True)
                    outputs = self._model(**inputs)
                    hidden = outputs.last_hidden_state.mean(dim=1)
                return self.proj(hidden)
            if isinstance(input_data, torch.Tensor) and input_data.dtype == torch.long:
                with torch.no_grad():
                    outputs = self._model(input_ids=input_data)
                    hidden = outputs.last_hidden_state.mean(dim=1)
                return self.proj(hidden)
        if isinstance(input_data, torch.Tensor) and input_data.dtype == torch.long:
            emb = self.embed(input_data)
            if emb.dim() == 3: emb = emb.mean(dim=1)
            elif emb.dim() == 2: emb = emb.mean(dim=0, keepdim=True)
            return self.proj(emb)
        if isinstance(input_data, torch.Tensor):
            return self.proj(input_data)
        return torch.zeros(1, self.input_dim)
