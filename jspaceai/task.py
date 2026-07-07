"""
玩具任务：连续时间序列预测

生成多模态时间序列——不同时段由不同的潜在规则驱动：
    时段 A（t mod 4 ∈ [0,1)）：正弦波 + 慢漂移
    时段 B（t mod 4 ∈ [1,2)）：快速振荡
    时段 C（t mod 4 ∈ [2,3)）：脉冲信号
    时段 D（t mod 4 ∈ [3,4)）：组合信号

任务：给定 x_t，预测 x_{t+1}。

设计意图：
    - 不同时段需要不同的"专家"处理——验证工作空间能否学到分工
    - 模式切换时需要工作空间广播给正确的专家——验证 J-space 路由
    - 扁平 MLP 在模式切换时会糊掉，工作空间架构应该更鲁棒
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ContinuousSequenceTask:
    """生成连续多模态时间序列"""

    def __init__(self, input_dim: int = 8, seq_len: int = 64, seed: int = 42):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        # 每个时段有自己的参数
        self.phase_len = 16  # 每个模式持续 16 步
        self.phase_params = self._gen_phase_params()

    def _gen_phase_params(self) -> list[dict]:
        """4 个时段，每段不同参数"""
        params = []
        for i in range(4):
            params.append({
                'freq': self.rng.uniform(0.1, 0.5),      # 振荡频率
                'phase': self.rng.uniform(0, 2 * np.pi),  # 相位
                'drift': self.rng.uniform(-0.02, 0.02),   # 慢漂移
                'amp': self.rng.uniform(0.3, 1.0),        # 振幅
                'noise': self.rng.uniform(0.01, 0.05),    # 噪声
            })
        return params

    def generate_sequence(self, n_steps: int) -> np.ndarray:
        """生成长度为 n_steps 的序列"""
        x = np.zeros((n_steps, self.input_dim), dtype=np.float32)
        state = np.zeros(self.input_dim, dtype=np.float32)

        for t in range(n_steps):
            phase_idx = (t // self.phase_len) % 4
            p = self.phase_params[phase_idx]

            # 每个维度独立演化，但共享相位
            for d in range(self.input_dim):
                d_offset = d * 0.3
                if phase_idx == 0:
                    # 正弦 + 漂移
                    state[d] = p['amp'] * np.sin(p['freq'] * t + p['phase'] + d_offset) + p['drift'] * t
                elif phase_idx == 1:
                    # 快速振荡
                    state[d] = p['amp'] * np.sin(p['freq'] * 3 * t + p['phase'] + d_offset)
                elif phase_idx == 2:
                    # 脉冲
                    pulse = 1.0 if (t % 5 == 0) else 0.0
                    state[d] = p['amp'] * pulse + 0.1 * np.sin(p['phase'] + d_offset)
                else:
                    # 组合
                    state[d] = (p['amp'] * 0.5 * np.sin(p['freq'] * t + p['phase'] + d_offset)
                                + p['amp'] * 0.3 * np.cos(p['freq'] * 2 * t + d_offset))

                state[d] += self.rng.normal(0, p['noise'])

            x[t] = state

        # 归一化到 [-1, 1]
        x = np.tanh(x)
        return x

    def generate_batch(self, batch_size: int, n_steps: int | None = None) -> torch.Tensor:
        """生成一个 batch 的序列"""
        n_steps = n_steps or self.seq_len
        batch = np.stack([self.generate_sequence(n_steps) for _ in range(batch_size)])
        return torch.from_numpy(batch)


class SequenceDataset(Dataset):
    """用于 DataLoader 的数据集"""

    def __init__(self, task: ContinuousSequenceTask, n_samples: int, seq_len: int):
        self.data = [task.generate_sequence(seq_len) for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])
