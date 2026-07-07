"""
训练器：预测学习目标 + 对比训练

学习目标（第一性原理）：
    智慧系统的核心是"内部建模世界"——即预测下一时刻世界状态。
    所以 loss = ||x_{t+1} - pred_t||²

    pred_t = model(x_{0:t})

这是自监督的——不需要标签，只需要时间序列本身。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .task import ContinuousSequenceTask


class Trainer:
    """通用训练器，适用于 JSpaceModel 和 FlatBaseline"""

    def __init__(self, model: nn.Module, task: ContinuousSequenceTask,
                 lr: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.history: list[float] = []

    def train_step(self, xs: torch.Tensor) -> float:
        """
        xs: (batch, T, input_dim)
        returns: loss value
        """
        xs = xs.to(self.device)
        # 预测目标：x_{t+1} = xs[:, 1:], 输入 xs[:, :-1]
        # 模型输出 preds[:, t] 应该预测 xs[:, t+1]
        preds, _ = self.model(xs)
        # preds: (batch, T, input_dim) —— preds[:, t] 是从 xs[:, :t+1] 预测的下一时刻
        # 对齐：preds[:, :-1] 预测 xs[:, 1:]
        pred = preds[:, :-1]
        target = xs[:, 1:]

        loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（ODE 训练容易爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate(self, xs: torch.Tensor) -> float:
        """评估，不更新参数"""
        self.model.eval()
        with torch.no_grad():
            xs = xs.to(self.device)
            preds, _ = self.model(xs)
            pred = preds[:, :-1]
            target = xs[:, 1:]
            loss = self.loss_fn(pred, target).item()
        self.model.train()
        return loss

    def train(self, n_steps: int = 500, batch_size: int = 32,
              eval_interval: int = 50, verbose: bool = True) -> list[float]:
        """完整训练循环"""
        self.model.train()
        for step in range(n_steps):
            xs = self.task.generate_batch(batch_size)
            loss = self.train_step(xs)
            self.history.append(loss)

            if verbose and (step + 1) % eval_interval == 0:
                eval_xs = self.task.generate_batch(64)
                eval_loss = self.evaluate(eval_xs)
                print(f"  step {step+1:4d} | train_loss {loss:.4f} | eval_loss {eval_loss:.4f}")

        return self.history

    def get_attention(self, xs: torch.Tensor) -> Optional[torch.Tensor]:
        """获取注意力权重（仅 JSpaceModel 有，用于可解释性）"""
        self.model.eval()
        with torch.no_grad():
            xs = xs.to(self.device)
            _, info = self.model(xs)
        self.model.train()
        return info.get('alpha', None)
