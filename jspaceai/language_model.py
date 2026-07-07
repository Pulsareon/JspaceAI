"""
语言版 JSpace 模型 + 自主进化机制

核心扩展：
    1. JSpaceLanguageModel: 在 JSpaceModel 基础上加 token embedding + logit 输出
    2. 在线学习：每个 forward 累积梯度并更新参数（边推理边学习）
    3. EWC（Elastic Weight Consolidation）：保护重要参数，防灾难性遗忘
    4. ExperienceReplay：经验回放缓冲区
    5. ExpertPlasticity：专家专业化追踪，新知识优先路由到"空闲"专家

数学形式：

    专家动力学（同 core.py）：
        dm_i/dt = -∇U_i(m_i) + J_i · w + P_i_in · embed(x)

    工作空间动力学（同 core.py）：
        τ_w · dw/dt = -w + Σ α_i · P_i_out(m_i)

    输出：
        logits = Q(w)  # 从工作空间投影到词汇表

    在线学习目标：
        L = -log p(x_{t+1} | x_{0:t}) + λ_EWC · Σ_i F_i · (θ_i - θ*_i)²

    EWC 中 F_i 是 Fisher 信息矩阵对角线，衡量参数重要性。
    重要参数被"锚定"在旧值附近，新知识只能修改不重要的参数。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import deque
import random

from .core import JSpaceConfig, Expert, JSpaceWorkspace


@dataclass
class LanguageConfig(JSpaceConfig):
    """语言模型配置，继承 JSpaceConfig"""
    vocab_size: int = 100           # 词汇表大小
    embed_dim: int = 16             # token embedding 维度（≠ input_dim，会投影）
    # input_dim 仍用 JSpaceConfig 的，作为工作空间接收的输入维度


class JSpaceLanguageModel(nn.Module):
    """
    语言版 JSpace 模型。

    流程：
        token → embedding → 投影到 input_dim → 喂入 JSpace 动力学
        工作空间 w → 投影到 vocab_size → logits → 采样 token

    每步可训练（在线学习）：
        forward 后用 cross-entropy loss 更新参数
    """

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        # 投影 embedding → input_dim（JSpace 期望的输入维度）
        self.input_proj = nn.Linear(config.embed_dim, config.input_dim, bias=False)

        # 专家池
        self.experts = nn.ModuleList([
            Expert(
                expert_dim=config.expert_dim,
                workspace_dim=config.workspace_dim,
                input_dim=config.input_dim,
                num_wells=config.num_wells,
                sparsity=config.jacobian_sparsity,
            )
            for _ in range(config.num_experts)
        ])

        # 工作空间
        self.workspace = JSpaceWorkspace(
            workspace_dim=config.workspace_dim,
            input_dim=config.input_dim,
            num_experts=config.num_experts,
        )

        # 输出头：w → logits
        self.output_head = nn.Sequential(
            nn.Linear(config.workspace_dim, 64),
            nn.Tanh(),
            nn.Linear(64, config.vocab_size),
        )

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            'w': torch.zeros(batch_size, self.config.workspace_dim, device=device),
            'm': [torch.zeros(batch_size, self.config.expert_dim, device=device)
                  for _ in range(self.config.num_experts)],
        }

    def step(self, state: dict, token_ids: torch.Tensor,
             record_trajectory: bool = False) -> tuple[dict, torch.Tensor, torch.Tensor, list]:
        """单时间步前向

        Args:
            state: {'w': ..., 'm': [...]}
            token_ids: (batch,) token indices
            record_trajectory: 是否记录 w 轨迹（J-lens 用）

        Returns:
            new_state, logits (batch, vocab_size), alpha (batch, num_experts),
            w_trajectory (list of (batch, workspace_dim)) 或空 list
        """
        w = state['w']
        ms = state['m']
        cfg = self.config
        w_trajectory = []

        # token → embedding → input projection
        emb = self.embedding(token_ids)        # (batch, embed_dim)
        x = self.input_proj(emb)               # (batch, input_dim)

        # ODE 子步积分（对应 Anthropic 论文的"层"）
        for substep in range(cfg.ode_steps):
            contributions = []
            new_ms = []
            for i, expert in enumerate(self.experts):
                m_next, contrib = expert(
                    ms[i], w, x,
                    dt=cfg.dt, noise_std=cfg.noise_std,
                )
                new_ms.append(m_next)
                contributions.append(contrib)
            contributions = torch.stack(contributions, dim=1)

            w, alpha = self.workspace(
                w, x, contributions,
                dt=cfg.dt, tau_w=cfg.tau_w,
            )
            ms = new_ms

            if record_trajectory:
                w_trajectory.append(w.detach())

        logits = self.output_head(w)  # (batch, vocab_size)
        new_state = {'w': w, 'm': ms}
        return new_state, logits, alpha, w_trajectory

    def forward(self, token_seqs: torch.Tensor, state: dict | None = None,
                record_trajectory: bool = False) -> tuple[torch.Tensor, dict]:
        """
        Args:
            token_seqs: (batch, T) token indices
            record_trajectory: 是否记录 w 轨迹

        Returns:
            logits: (batch, T, vocab_size)
            info: {'alpha': (batch, T, num_experts), 'w_norm': (batch, T),
                   'w_trajectory': list of (batch, T, workspace_dim) 或空}
        """
        batch_size, T = token_seqs.shape
        device = token_seqs.device

        if state is None:
            state = self.init_state(batch_size, device)

        logits_list = []
        alphas = []
        w_norms = []
        w_traj_per_step = []  # list of (list of substep w)
        for t in range(T):
            state, logits, alpha, w_traj = self.step(
                state, token_seqs[:, t], record_trajectory=record_trajectory
            )
            logits_list.append(logits)
            alphas.append(alpha)
            w_norms.append(state['w'].norm(dim=-1))
            if record_trajectory and w_traj:
                w_traj_per_step.append(torch.stack(w_traj, dim=1))  # (batch, n_substeps, workspace_dim)

        logits = torch.stack(logits_list, dim=1)  # (batch, T, vocab_size)
        info = {
            'alpha': torch.stack(alphas, dim=1),     # (batch, T, num_experts)
            'w_norm': torch.stack(w_norms, dim=1),    # (batch, T)
        }
        if record_trajectory and w_traj_per_step:
            info['w_trajectory'] = torch.stack(w_traj_per_step, dim=1)  # (batch, T, n_substeps, workspace_dim)
        return logits, info

    @torch.no_grad()
    def generate(self, prompt: list[int], n_new: int = 50, temperature: float = 1.0,
                 top_k: int = 5) -> list[int]:
        """自回归生成

        Args:
            prompt: 起始 token ids
            n_new: 生成的新 token 数
            temperature: 采样温度
            top_k: top-k 采样
        """
        self.eval()
        device = next(self.parameters()).device
        state = self.init_state(1, device)

        # 预热 state with prompt
        tokens = list(prompt)
        for tok in tokens:
            state, _, _, _ = self.step(state, torch.tensor([tok], device=device))

        # 生成
        generated = []
        for _ in range(n_new):
            state, logits, _, _ = self.step(state, torch.tensor([tokens[-1]], device=device))
            logits = logits[0] / max(temperature, 1e-6)

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                vals, idxs = logits.topk(top_k)
                probs = F.softmax(vals, dim=-1)
                next_tok = idxs[torch.multinomial(probs, 1)].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

            generated.append(next_tok)
            tokens.append(next_tok)

        self.train()
        return generated


class ExperienceReplay:
    """
    经验回放缓冲区。

    存储见过的序列片段，训练时随机采样混入当前 batch。
    防止灾难性遗忘——旧知识被定期"复习"。
    """

    def __init__(self, capacity: int = 1000, seq_len: int = 64):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, token_seq: torch.Tensor):
        """push 一个序列 (T,) 或 (batch, T)"""
        if token_seq.dim() == 1:
            token_seq = token_seq.unsqueeze(0)
        for seq in token_seq:
            if len(seq) >= self.seq_len:
                self.buffer.append(seq.clone())

    def sample(self, batch_size: int) -> torch.Tensor | None:
        """采样 (batch_size, seq_len)"""
        if len(self.buffer) < batch_size:
            return None
        samples = random.sample(list(self.buffer), batch_size)
        # 随机裁剪到 seq_len
        result = []
        for s in samples:
            if len(s) > self.seq_len:
                start = random.randint(0, len(s) - self.seq_len - 1)
                result.append(s[start:start + self.seq_len])
            else:
                result.append(s)
        return torch.stack(result)


class EWCOptimizer:
    """
    Elastic Weight Consolidation 优化器包装。

    核心思想：参数 θ 有"重要性" F（Fisher 信息）。
    重要参数偏离原值 θ* 会被惩罚。
    新知识只能修改不重要的参数。

    L_total = L_task + λ · Σ_i F_i · (θ_i - θ*_i)²

    工作流：
        1. 正常训练一段时间
        2. 调用 consolidate()：计算 Fisher 信息，锚定当前参数
        3. 继续训练——loss 中加入 EWC 正则
        4. 周期性 consolidate（更新锚点和重要性）
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3,
                 ewc_lambda: float = 1.0, max_grad_norm: float = 1.0):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ewc_lambda = ewc_lambda
        self.max_grad_norm = max_grad_norm

        # Fisher 信息和锚定参数
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchored_params: dict[str, torch.Tensor] = {}

    def consolidate(self, data_sample: torch.Tensor, n_samples: int = 50):
        """计算 Fisher 信息并锚定当前参数

        Args:
            data_sample: (batch, T) 用于计算 Fisher 的数据样本
            n_samples: 采样次数（Fisher 信息的 Monte Carlo 估计）
        """
        # 保存当前参数作为锚点
        self.anchored_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        # 计算 Fisher 信息（对角近似）
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        self.model.eval()
        for _ in range(n_samples):
            self.model.zero_grad()
            logits, _ = self.model(data_sample)
            # 只用 logits[:, :-1] 对应 targets[:, 1:] 的部分
            logits_pred = logits[:, :-1]  # (batch, T-1, vocab)
            probs = F.softmax(logits_pred, dim=-1)  # (batch, T-1, vocab)
            # 采样 token 计算 Fisher
            sampled_tokens = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), 1
            ).view_as(logits_pred[:, :, 0])  # (batch, T-1)
            log_probs = F.log_softmax(logits_pred, dim=-1)
            loss = -log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).mean()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        # 平均
        for name in fisher:
            fisher[name] /= n_samples

        self.fisher = fisher
        self.model.zero_grad()
        self.model.train()

    def ewc_penalty(self) -> torch.Tensor:
        """计算 EWC 正则项"""
        if not self.fisher:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                penalty = penalty + (self.fisher[name] * (param - self.anchored_params[name]) ** 2).sum()
        return penalty

    def step(self, loss: torch.Tensor):
        """一步优化：task loss + EWC 正则"""
        total_loss = loss + self.ewc_lambda * self.ewc_penalty()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return total_loss.item()


class ExpertPlasticity:
    """
    专家结构可塑性追踪。

    追踪每个专家的"专业化程度"：
        - 哪些专家在处理哪些模式
        - 哪些专家"负载过重"（应该分裂或新增）
        - 哪些专家"空闲"（可以接收新知识）

    这不是真正的动态增删专家（实现复杂），而是：
        - 统计专家使用率
        - 在路由时给空闲专家加权（鼓励新知识流向空闲专家）
    """

    def __init__(self, num_experts: int, ema_alpha: float = 0.99):
        self.num_experts = num_experts
        self.ema_alpha = ema_alpha
        # 每个专家的使用率（EMA）
        self.usage = torch.ones(num_experts) / num_experts
        # 每个专家的"领地"——它擅长的 token 分布
        self.expert_specialization: list[dict[int, float]] = [{} for _ in range(num_experts)]

    def update(self, alpha: torch.Tensor, tokens: torch.Tensor):
        """更新专家统计

        Args:
            alpha: (batch, T, num_experts) 注意力权重
            tokens: (batch, T) 对应的 token
        """
        # 使用率（时间维度平均）
        usage_batch = alpha.mean(dim=(0, 1)).detach().cpu()  # (num_experts,)
        self.usage = self.ema_alpha * self.usage + (1 - self.ema_alpha) * usage_batch

        # 专业化：每个专家最常处理哪些 token
        alpha_flat = alpha.reshape(-1, self.num_experts).detach().cpu()  # (batch*T, num_experts)
        tokens_flat = tokens.reshape(-1).detach().cpu().tolist()
        for tok, weights in zip(tokens_flat, alpha_flat):
            for i, w in enumerate(weights.tolist()):
                if w > 0.1:  # 只记录显著激活
                    self.expert_specialization[i][tok] = \
                        self.expert_specialization[i].get(tok, 0) + w

    def get_diversity_bonus(self) -> torch.Tensor:
        """返回多样性奖励——给使用率低的专家加权

        在路由注意力上加上这个 bonus，鼓励新知识流向空闲专家
        """
        # 使用率越低，bonus 越高
        bonus = (1.0 - self.usage) / self.usage.clamp(min=1e-4)
        bonus = bonus / bonus.sum()  # 归一化
        return bonus

    def get_stats(self) -> dict:
        """返回可解释性统计"""
        return {
            'usage': self.usage.tolist(),
            'top_specialization': [
                sorted(s.items(), key=lambda x: -x[1])[:5]
                for s in self.expert_specialization
            ],
        }
