"""
Jacobian Lens (J-lens) —— 观测模型内部 J-space 的可解释性工具

灵感来自 Anthropic 2026 论文 "Verbalizable Representations Form a Global
Workspace in Language Models"。

核心思想：
    J-lens 计算中间层激活对最终输出的平均因果效应。
    J_ℓ = E[∂h_final / ∂h_ℓ]  —— 跨大量 context 平均的 Jacobian
    lens(h_ℓ) = softmax(W_U · norm(J_ℓ · h_ℓ))

    J-lens 向量 = W_U · J_ℓ 的行，每个向量对应词汇表中的一个 token。
    一个激活向量在 J-lens 下的 top tokens = 模型"准备要说"的概念。

与 logit lens 的区别：
    logit lens 直接用 W_U 投影（假设 J_ℓ = I）。
    J-lens 修正了层间表征变化，能在更早的层揭示可解释内容。

在我们的 ODE 架构中：
    - 每个 ODE 子步是一个"层"
    - workspace w 在每个子步演化
    - J-lens 可以在任意子步读 w，揭示模型在该时刻"在想什么"

实现简化：
    - 完整 J-lens 需要 backprop 从输出到中间层，在我们的 ODE 模型中代价高
    - 我们用近似：直接训练一个 lens matrix L_ℓ，让 lens(h) ≈ output
    - L_ℓ 通过最小化 ||output - L_ℓ · h||² 在数据上学习
    - 这等价于 tuned lens，但在我们的架构中更高效
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JLensConfig:
    """J-lens 配置"""
    n_substeps: int = 4              # ODE 子步数（对应"层"数）
    workspace_dim: int = 32          # 工作空间维度
    vocab_size: int = 100            # 词汇表大小
    top_k: int = 10                  # 默认 top-k 读出


class JLensProbe(nn.Module):
    """
    单个子步的 J-lens 探针。

    学习一个线性映射 L: workspace_dim → vocab_size，
    使得 lens(w) ≈ model_output。

    这近似了 J_ℓ · W_U（Jacobian 与 unembedding 的复合）。
    """

    def __init__(self, workspace_dim: int, vocab_size: int):
        super().__init__()
        self.lens = nn.Linear(workspace_dim, vocab_size, bias=False)
        # 用 output_head 的权重初始化（如果可用）
        nn.init.normal_(self.lens.weight, std=0.02)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w → logits (近似 J-lens 读出)"""
        return self.lens(w)

    def top_tokens(self, w: torch.Tensor, idx_to_char: dict,
                   top_k: int = 10) -> list[tuple[str, float]]:
        """获取 top-k token 及其概率"""
        logits = self.forward(w)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(top_k)
        return [(idx_to_char.get(i.item(), '?'), p.item())
                for i, p in zip(topk_idx[0], topk_probs[0])]


class JLensSuite(nn.Module):
    """
    J-lens 套件：每个 ODE 子步一个探针。

    在 forward 时记录每个子步的 w，用对应探针读出。
    训练时让每个探针预测最终输出。

    结构对应 Anthropic 论文的三层分层：
        子步 0-0:  sensory（输入处理，J-lens 噪声大）
        子步 1-2:  workspace（抽象思考，J-lens 可解释）
        子步 3:    motor（输出准备，J-lens ≈ output）
    """

    def __init__(self, config: JLensConfig):
        super().__init__()
        self.config = config
        self.probes = nn.ModuleList([
            JLensProbe(config.workspace_dim, config.vocab_size)
            for _ in range(config.n_substeps)
        ])

    def forward(self, w_trajectory: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        对轨迹中每个 w 读出 logits

        Args:
            w_trajectory: list of (batch, workspace_dim)，每个子步的 w

        Returns:
            list of (batch, vocab_size)，每个子步的 lens 读出
        """
        return [probe(w) for probe, w in zip(self.probes, w_trajectory)]

    def train_on_trajectory(self, w_trajectory: list[torch.Tensor],
                            target_ids: torch.Tensor) -> float:
        """
        训练探针：让每个子步的 lens 读出预测最终 target token

        Args:
            w_trajectory: 每个子步的 w
            target_ids: 目标 token ids (batch,)

        Returns:
            平均 loss
        """
        total_loss = 0.0
        for probe, w in zip(self.probes, w_trajectory):
            pred = probe(w)
            loss = F.cross_entropy(pred, target_ids)
            total_loss += loss
        return total_loss / len(w_trajectory)


class WorkspaceAblator:
    """
    Workspace ablation 工具——验证 selectivity。

    ablate workspace 后看哪些能力受损：
        - 自动任务（续写、分类）应该不受影响
        - 灵活推理（多跳、规划）应该受损

    实现：在 forward 时把 w 的 top-k J-lens 方向投影掉。
    """

    def __init__(self, model, lens_suite: JLensSuite):
        self.model = model
        self.lens_suite = lens_suite

    @torch.no_grad()
    def get_top_lens_directions(self, w: torch.Tensor, k: int = 5) -> torch.Tensor:
        """获取 w 当前激活最强的 k 个 J-lens 方向"""
        # 用第一个 workspace 探针（中间子步）
        probe = self.lens_suite.probes[len(self.lens_suite.probes) // 2]
        logits = probe(w)  # (batch, vocab)
        # top-k token 的 lens 向量（probe.lens.weight 的行）
        topk_vals, topk_idx = logits.topk(k, dim=-1)  # (batch, k)
        # 获取这些 token 对应的 lens 方向
        # probe.lens.weight: (vocab, workspace_dim)
        directions = probe.lens.weight[topk_idx]  # (batch, k, workspace_dim)
        return directions

    def ablate_workspace(self, w: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Ablate workspace 的 top-k J-lens 方向

        把 w 在这些方向上的投影去掉，保留正交分量。
        """
        directions = self.get_top_lens_directions(w, k)  # (batch, k, workspace_dim)
        # 对每个方向，投影掉
        w_ablated = w.clone()
        for b in range(w.shape[0]):
            for d in directions[b]:  # (workspace_dim,)
                d_norm = d / (d.norm() + 1e-8)
                proj = (w_ablated[b] @ d_norm) * d_norm
                w_ablated[b] = w_ablated[b] - proj
        return w_ablated


class DirectedModulation:
    """
    Directed Modulation——让模型被指令"想某概念"。

    实现：在 forward 时给 workspace w 注入一个概念向量。
    这个向量从 J-lens 的某个 token 方向获取。

    对应 Anthropic 论文实验：
        "concentrate on citrus fruits" → orange 出现在 J-lens
        即使输出在抄无关文本，workspace 里装的是被指令的概念。

    机制：
        1. 获取概念 token 的 J-lens 向量 v_concept
        2. 在 forward 时给 w 加上 α · v_concept
        3. 模型输出会受这个注入影响
    """

    def __init__(self, model, lens_suite: JLensSuite):
        self.model = model
        self.lens_suite = lens_suite

    def get_concept_vector(self, token_id: int, substep: int = None) -> torch.Tensor:
        """获取某个 token 在 J-lens 中的方向向量"""
        if substep is None:
            substep = len(self.lens_suite.probes) // 2
        probe = self.lens_suite.probes[substep]
        # probe.lens.weight: (vocab, workspace_dim)
        return probe.lens.weight[token_id].clone()

    def modulate_state(self, state: dict, token_id: int,
                       strength: float = 1.0, substep: int = None) -> dict:
        """给 state 的 workspace 注入概念向量"""
        concept_vec = self.get_concept_vector(token_id, substep)  # (workspace_dim,)
        new_state = {
            'w': state['w'] + strength * concept_vec.unsqueeze(0),  # (1, workspace_dim)
            'm': state['m'],
        }
        return new_state


class CounterfactualReflection:
    """
    Counterfactual Reflection Training——通过塑造 J-space 来塑造行为。

    灵感：Anthropic 论文发现，训练模型"如果被打断要反思什么原则"，
    会让它在正常工作时也遵守这些原则——因为训练塑造了 J-space 内容。

    实现：
        1. 正常 forward 产生输出
        2. 在输出后追加"反思提示"（如"反思：我应该..."）
        3. 让模型在反思提示下生成反思内容
        4. 用反思内容的 loss 反向传播，更新模型

    这让模型的 J-space 在相关 context 下自然装载这些原则。
    """

    def __init__(self, model, tokenizer, reflection_prompt: str = "\nReflect: "):
        self.model = model
        self.tokenizer = tokenizer
        self.reflection_prompt_ids = tokenizer.encode(reflection_prompt)

    def create_reflection_sequence(self, input_ids: list[int],
                                   reflection_target_ids: list[int]) -> list[int]:
        """创建反思训练序列：input + reflection_prompt + reflection_target"""
        return input_ids + self.reflection_prompt_ids + reflection_target_ids

    def compute_reflection_loss(self, input_seq: torch.Tensor,
                                reflection_target: torch.Tensor) -> torch.Tensor:
        """
        计算反思训练 loss

        Args:
            input_seq: (batch, T) 包含 input + reflection_prompt
            reflection_target: (batch, T_reflect) 期望的反思内容

        Returns:
            loss
        """
        # forward 整个序列
        logits, _ = self.model(input_seq)

        # 反思部分是序列末尾的 reflection_target 长度
        T_reflect = reflection_target.shape[1]
        reflect_logits = logits[:, -T_reflect:]  # (batch, T_reflect, vocab)

        return F.cross_entropy(
            reflect_logits.reshape(-1, self.model.config.vocab_size),
            reflection_target.reshape(-1),
        )
