"""
对比基线：扁平网络（无工作空间、无专家分块、无动力学）

用同样的参数量预算，验证"工作空间 + J-space 广播"架构比扁平 MLP 好。
这是关键对比——如果工作空间架构赢不了同样大小的 MLP，那架构本身没意义。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlatBaseline(nn.Module):
    """
    扁平 MLP 基线：参数量与 JSpaceModel 相当，但无结构。

    - 无专家分块（单一隐层）
    - 无工作空间（无广播机制）
    - 无动力学（每步独立预测，无状态传递）
    - 无 J-space 路由

    这代表"用同样算力做暴力拟合"的路线。
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 90, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xs: torch.Tensor, state=None) -> tuple[torch.Tensor, dict]:
        """
        xs: (batch, T, input_dim)
        returns: preds (batch, T, input_dim), info (空)
        """
        preds = self.net(xs)  # (batch, T, input_dim)
        return preds, {}
