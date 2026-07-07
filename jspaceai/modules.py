"""
外挂模块系统 —— 可热插拔的外部能力

设计：
    - 核心心智不依赖外挂，断开后继续工作
    - 标准接口，任何模块都能插入
    - 运行时热插拔，不需要重启
    - 心智知道外挂状态
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np, time, math
from typing import Optional, Any
from abc import ABC, abstractmethod


class ExternalModule(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def disconnect(self): ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def query(self, input_data: Any) -> Any: ...
    @abstractmethod
    def describe(self) -> str: ...


class SmallModelModule(ExternalModule):
    """小模型外挂——提供语言/知识能力。可热插拔。"""

    def __init__(self, workspace_dim=64, model_name="placeholder"):
        self._name = f"small_model:{model_name}"
        self.workspace_dim = workspace_dim
        self.model_name = model_name
        self._connected = False
        self._model = None
        self._tokenizer = None
        self._proj = None

    @property
    def name(self): return self._name

    def connect(self) -> bool:
        try:
            from transformers import AutoModel, AutoTokenizer
            print(f"  [外挂] 加载 {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            hs = self._model.config.hidden_size
            self._proj = nn.Linear(hs, self.workspace_dim, bias=False)
            self._connected = True
            print(f"  [外挂] {self.model_name} 已连接 (hidden={hs})")
            return True
        except ImportError:
            print(f"  [外挂] transformers 未装，占位模式")
            self._connected = True
            return True
        except Exception as e:
            print(f"  [外挂] 连接失败: {e}")
            return False

    def disconnect(self):
        if self._model is not None:
            del self._model, self._tokenizer
            self._model = self._tokenizer = None
        self._proj = None
        self._connected = False
        print(f"  [外挂] {self.name} 已断开")

    def is_connected(self): return self._connected

    def query(self, input_data):
        if not self._connected:
            return None
        if self._model is None:
            n = input_data.shape[0] if isinstance(input_data, torch.Tensor) and input_data.dim() > 0 else 1
            return torch.randn(n, self.workspace_dim)
        try:
            text = str(input_data) if not isinstance(input_data, torch.Tensor) \
                else f"state_{input_data.mean().item():.3f}"
            with torch.no_grad():
                inputs = self._tokenizer(text, return_tensors="pt",
                                        truncation=True, max_length=128)
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state.mean(dim=1)
                return self._proj(hidden)
        except Exception as e:
            return None

    def describe(self):
        if self._model is None and self._connected:
            return "占位模式"
        return f"语言模型 {self.model_name}"


class KnowledgeBaseModule(ExternalModule):
    """知识库外挂——向量检索"""

    def __init__(self, workspace_dim=64, capacity=10000):
        self._name = "knowledge_base"
        self.workspace_dim = workspace_dim
        self.capacity = capacity
        self._connected = False
        self.entries = []
        self.vectors = None

    @property
    def name(self): return self._name

    def connect(self):
        self._connected = True
        print(f"  [外挂] 知识库已连接 ({len(self.entries)} 条)")
        return True

    def disconnect(self):
        self._connected = False
        print(f"  [外挂] 知识库已断开（数据保留）")

    def is_connected(self): return self._connected

    def add_entry(self, vector, text, metadata=None):
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append({'vector': np.array(vector), 'text': text,
                           'metadata': metadata or {}})
        self.vectors = np.array([e['vector'] for e in self.entries])

    def query(self, input_data):
        if not self._connected or not self.entries:
            return []
        if not isinstance(input_data, torch.Tensor):
            return []
        q = input_data[0].cpu().numpy() if input_data.dim() > 1 else input_data.cpu().numpy()
        if self.vectors is None or len(self.vectors) == 0:
            return []
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q)
        sims = self.vectors @ q / (norms + 1e-8)
        top = np.argsort(sims)[-5:][::-1]
        return [{'text': self.entries[i]['text'], 'sim': float(sims[i])} for i in top]

    def describe(self):
        return f"知识库（{len(self.entries)}/{self.capacity}）"


class ToolModule(ExternalModule):
    """工具外挂——可调用的外部工具"""

    def __init__(self):
        self._name = "tools"
        self._connected = False
        self.tools = {}

    @property
    def name(self): return self._name

    def connect(self):
        self._connected = True
        self.register('calc', lambda e: eval(e, {'__builtins__': {}}, {'math': math}), "计算")
        self.register('time', lambda: time.time(), "时间戳")
        print(f"  [外挂] 工具箱已连接 ({len(self.tools)} 工具)")
        return True

    def disconnect(self):
        self._connected = False
        self.tools.clear()
        print(f"  [外挂] 工具箱已断开")

    def is_connected(self): return self._connected

    def register(self, name, func, desc=""):
        self.tools[name] = {'func': func, 'desc': desc}

    def query(self, input_data):
        if not self._connected:
            return None
        if isinstance(input_data, dict) and 'tool' in input_data:
            tn = input_data['tool']
            args = input_data.get('args', [])
            if tn in self.tools:
                try:
                    return self.tools[tn]['func'](*args)
                except Exception as e:
                    return f"错误: {e}"
        return None

    def describe(self):
        return f"工具箱（{list(self.tools.keys())}）"


class ModuleDock:
    """外挂坞——USB hub 式管理可热插拔模块。

    用法：
        dock = ModuleDock()
        dock.register('llm', SmallModelModule())
        dock.connect('llm')          # 热插上
        result = dock.query('llm', "hello")
        dock.disconnect('llm')       # 拔掉，心智不停
    """

    def __init__(self, workspace_dim=64):
        self.workspace_dim = workspace_dim
        self.slots = {}
        self.log = []

    def register(self, name, module):
        self.slots[name] = module

    def connect(self, name) -> bool:
        if name not in self.slots:
            return False
        if self.slots[name].is_connected():
            return True
        ok = self.slots[name].connect()
        self.log.append({'time': time.time(), 'slot': name,
                        'action': 'connect', 'ok': ok})
        return ok

    def disconnect(self, name):
        if name not in self.slots:
            return
        self.slots[name].disconnect()
        self.log.append({'time': time.time(), 'slot': name,
                        'action': 'disconnect', 'ok': True})

    def disconnect_all(self):
        for n in list(self.slots):
            if self.slots[n].is_connected():
                self.disconnect(n)

    def is_connected(self, name) -> bool:
        m = self.slots.get(name)
        return m.is_connected() if m else False

    def query(self, name, input_data):
        """查询某个外挂。未连接返回 None。"""
        m = self.slots.get(name)
        if m and m.is_connected():
            return m.query(input_data)
        return None

    def status(self) -> dict:
        """所有外挂状态"""
        return {name: {
            'connected': m.is_connected(),
            'description': m.describe(),
        } for name, m in self.slots.items()}

    def connected_count(self) -> int:
        return sum(1 for m in self.slots.values() if m.is_connected())
