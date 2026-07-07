"""
核心架构：专家模块 + J-space 工作空间 + ODE 动力学

数学形式（每个 forward 时间步内做 Euler 积分若干子步）：

    专家 i 的状态 m_i:
        dm_i/dt = -∇U_i(m_i) + J_i · w + P_i_in · x

    U_i(m_i) = ½ ||m_i||² - ½ Σ_k softplus(a_ik · m_ik + b_ik)
    （多井势能：阻尼项 + softplus 形成的局部吸引子）

    工作空间 w:
        τ_w · dw/dt = -w + Σ_i α_i · P_i_out(m_i)
        α_i = softmax(<q, P_i_out(m_i)>)  q = MLP(x, w)

    Jacobian 路由 J_i: 稀疏线性映射，每个专家只对 w 的少数维度敏感
    输出门控：当 ||w|| > θ 时触发输出 R(w)

所有参数都可 backprop 训练。学习目标是预测下一时刻的输入。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class JSpaceConfig:
    """模型超参，全部可调"""
    input_dim: int = 8              # 输入 x 的维度
    workspace_dim: int = 32         # 工作空间 w 的维度（J-space）
    expert_dim: int = 16            # 每个专家内部状态 m_i 的维度
    num_experts: int = 5            # 专家数量
    num_wells: int = 4              # 每个专家势能景观的井数
    ode_steps: int = 4              # 每个时间步内 ODE 积分子步数
    dt: float = 0.1                 # ODE 积分步长
    tau_w: float = 0.3              # 工作空间时间常数
    output_threshold: float = 0.5   # 输出门控阈值（船舶涌出阈值）
    jacobian_sparsity: int = 8      # 每个 J_i 只保留前 k 大的连接
    noise_std: float = 0.01         # 内部噪声 ξ(t) 的标准差


class Expert(nn.Module):
    """
    单个专家模块。

    状态：m_i ∈ R^{expert_dim}
    势能：U_i(m_i) = ½||m_i||² - ½ Σ_k softplus(a_k · m_i + b_k) · w_k
        - ½||m_i||² 是阻尼项（拉回原点）
        - softplus 项创造多个局部吸引子（多井势能 → 内部"思考"）
    动力学：dm_i/dt = -∇U_i(m_i) + J_i · w + P_in · x + ξ

    J_i 是稀疏 Jacobian：从工作空间 w 路由信息进来。
    """

    def __init__(self, expert_dim: int, workspace_dim: int, input_dim: int,
                 num_wells: int, sparsity: int):
        super().__init__()
        self.expert_dim = expert_dim
        self.workspace_dim = workspace_dim
        self.num_wells = num_wells

        # 势能景观参数：每个井是一个 softplus 形成的吸引子
        # U(m) = 0.5||m||^2 - 0.5 * sum_k softplus(a_k @ m + b_k)
        # ∇U(m) = m - 0.5 * sum_k sigmoid(a_k @ m + b_k) * a_k
        self.well_a = nn.Parameter(torch.randn(num_wells, expert_dim) * 0.3)
        self.well_b = nn.Parameter(torch.zeros(num_wells))

        # P_in: 输入投影 x -> m_i 的扰动
        self.P_in = nn.Linear(input_dim, expert_dim, bias=False)

        # P_out: 模块输出到工作空间的投影
        self.P_out = nn.Linear(expert_dim, workspace_dim, bias=False)

        # J_i: 稀疏 Jacobian，从 w 路由信息到 m_i
        # 用 top-k 稀疏：训练时学习一个 full matrix，但只激活 top-k
        self.J_raw = nn.Parameter(torch.randn(expert_dim, workspace_dim) * 0.1)
        self.sparsity = sparsity

        # 注意：sparsity 通过 forward 时 top-k 选择实现，可微性通过稀疏 mask 保留

    def get_sparse_J(self) -> torch.Tensor:
        """获取稀疏化的 Jacobian：每行只保留 top-k 元素"""
        if self.sparsity >= self.workspace_dim:
            return self.J_raw
        # 对每行做 top-k（按绝对值）
        abs_J = self.J_raw.abs()
        topk_vals, topk_idx = abs_J.topk(self.sparsity, dim=-1)
        mask = torch.zeros_like(self.J_raw)
        mask.scatter_(-1, topk_idx, 1.0)
        return self.J_raw * mask

    def grad_potential(self, m: torch.Tensor) -> torch.Tensor:
        """计算势能梯度 ∇U_i(m)
        U(m) = 0.5||m||^2 - 0.5 * sum_k softplus(a_k @ m + b_k)
        ∇U(m) = m - 0.5 * sum_k sigmoid(a_k @ m + b_k) * a_k
        """
        # m: (batch, expert_dim)
        # well_a: (num_wells, expert_dim)
        # a_k @ m: (batch, num_wells)
        am = F.linear(m, self.well_a, self.well_b)  # (batch, num_wells)
        sig = torch.sigmoid(am)  # (batch, num_wells)
        # sum_k sigmoid(...) * a_k: (batch, expert_dim)
        # well_a: (num_wells, expert_dim), sig: (batch, num_wells) -> (batch, 1, num_wells)
        # 用 matmul: sig @ well_a -> (batch, expert_dim)
        grad_wells = torch.matmul(sig, self.well_a)  # (batch, expert_dim)
        return m - 0.5 * grad_wells

    def forward(self, m: torch.Tensor, w: torch.Tensor, x: torch.Tensor,
                dt: float, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
        """一步 ODE 积分（Euler 法）

        Args:
            m: (batch, expert_dim) 当前状态
            w: (batch, workspace_dim) 工作空间状态
            x: (batch, input_dim) 输入
            dt: 步长
            noise_std: 噪声标准差

        Returns:
            m_next: (batch, expert_dim) 下一状态
            contribution: (batch, workspace_dim) 对工作空间的贡献（pre-attention）
        """
        # 动力学: dm/dt = -∇U(m) + J·w + P_in·x + ξ
        J = self.get_sparse_J()  # (expert_dim, workspace_dim)
        w_proj = F.linear(w, J)  # (batch, expert_dim)
        x_proj = self.P_in(x)    # (batch, expert_dim)
        grad_U = self.grad_potential(m)  # (batch, expert_dim)

        noise = torch.randn_like(m) * noise_std if noise_std > 0 else 0.0

        dm = -grad_U + w_proj + x_proj + noise
        m_next = m + dt * dm

        # 对工作空间的贡献
        contribution = self.P_out(m_next)  # (batch, workspace_dim)
        return m_next, contribution


class JSpaceWorkspace(nn.Module):
    """
    全局工作空间 w。

    动力学: τ_w · dw/dt = -w + Σ_i α_i · P_i_out(m_i)
    α_i = softmax(<q, P_i_out(m_i)>)  q = MLP(x, w)

    这个 α_i 是"注意力"——决定哪个专家的内容进入工作空间。
    """

    def __init__(self, workspace_dim: int, input_dim: int, num_experts: int):
        super().__init__()
        self.workspace_dim = workspace_dim

        # Query 生成器：从 (x, w) 生成 query 向量
        self.query_gen = nn.Sequential(
            nn.Linear(input_dim + workspace_dim, 32),
            nn.Tanh(),
            nn.Linear(32, workspace_dim),
        )

    def forward(self, w: torch.Tensor, x: torch.Tensor,
                contributions: torch.Tensor, dt: float,
                tau_w: float) -> tuple[torch.Tensor, torch.Tensor]:
        """一步工作空间演化

        Args:
            w: (batch, workspace_dim)
            x: (batch, input_dim)
            contributions: (batch, num_experts, workspace_dim) 各专家的贡献
            dt: 步长
            tau_w: 时间常数

        Returns:
            w_next: (batch, workspace_dim)
            alpha: (batch, num_experts) 注意力权重（可解释性用）
        """
        # 生成 query
        q = self.query_gen(torch.cat([x, w], dim=-1))  # (batch, workspace_dim)

        # 计算每个专家的注意力分数
        # contributions: (batch, num_experts, workspace_dim)
        # q: (batch, workspace_dim) -> (batch, 1, workspace_dim)
        scores = (contributions * q.unsqueeze(1)).sum(dim=-1)  # (batch, num_experts)
        alpha = F.softmax(scores, dim=-1)  # (batch, num_experts)

        # 加权聚合
        # alpha: (batch, num_experts, 1) * contributions: (batch, num_experts, workspace_dim)
        aggregated = (alpha.unsqueeze(-1) * contributions).sum(dim=1)  # (batch, workspace_dim)

        # 动力学: τ_w · dw/dt = -w + aggregated
        dw = (-w + aggregated) / tau_w
        w_next = w + dt * dw
        return w_next, alpha


class JSpaceModel(nn.Module):
    """
    完整模型：N 个专家 + 工作空间 + 输出门控 + 预测头。

    forward 流程（每个时间步）：
        1. 每个专家从 (m_i, w, x) 更新 m_i，产出对工作空间的贡献
        2. 工作空间从 (w, x, contributions) 更新 w
        3. （可选）当 ||w|| > threshold 时输出 R(w)
        4. 预测头 Q(w) 预测下一时刻输入

    时间序列处理：对长度 T 的输入序列，依次跑 T 步，返回每步的预测。
    """

    def __init__(self, config: JSpaceConfig):
        super().__init__()
        self.config = config

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

        self.workspace = JSpaceWorkspace(
            workspace_dim=config.workspace_dim,
            input_dim=config.input_dim,
            num_experts=config.num_experts,
        )

        # 输出门控：R(w) → action（这里 action = 预测的下一时刻输入）
        self.predictor = nn.Sequential(
            nn.Linear(config.workspace_dim, 32),
            nn.Tanh(),
            nn.Linear(32, config.input_dim),
        )

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """初始化内部状态"""
        return {
            'w': torch.zeros(batch_size, self.config.workspace_dim, device=device),
            'm': [torch.zeros(batch_size, self.config.expert_dim, device=device)
                  for _ in range(self.config.num_experts)],
        }

    def step(self, state: dict, x: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单时间步前向

        Returns:
            new_state: 更新后的状态
            pred: (batch, input_dim) 预测的下一时刻输入
            alpha: (batch, num_experts) 注意力权重（可解释性）
            w_norm: (batch,) 工作空间范数（输出门控信号）
        """
        w = state['w']
        ms = state['m']
        cfg = self.config

        # ODE 子步积分
        for _ in range(cfg.ode_steps):
            # 1. 每个专家更新
            contributions = []
            new_ms = []
            for i, expert in enumerate(self.experts):
                m_next, contrib = expert(
                    ms[i], w, x,
                    dt=cfg.dt, noise_std=cfg.noise_std,
                )
                new_ms.append(m_next)
                contributions.append(contrib)
            contributions = torch.stack(contributions, dim=1)  # (batch, num_experts, workspace_dim)

            # 2. 工作空间更新
            w, alpha = self.workspace(
                w, x, contributions,
                dt=cfg.dt, tau_w=cfg.tau_w,
            )
            ms = new_ms

        # 3. 输出：预测下一时刻输入（船舶涌出，但这里为了训练简化为每步都预测）
        pred = self.predictor(w)
        w_norm = w.norm(dim=-1)

        new_state = {'w': w, 'm': ms}
        return new_state, pred, alpha, w_norm

    def forward(self, xs: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            xs: (batch, T, input_dim) 输入序列
            state: 初始状态，None 则初始化

        Returns:
            preds: (batch, T, input_dim) 每步对下一时刻的预测
            info: 包含注意力、w_norm 等可解释性信息
        """
        batch_size, T, _ = xs.shape
        device = xs.device

        if state is None:
            state = self.init_state(batch_size, device)

        preds = []
        alphas = []
        w_norms = []
        for t in range(T):
            state, pred, alpha, w_norm = self.step(state, xs[:, t])
            preds.append(pred)
            alphas.append(alpha)
            w_norms.append(w_norm)

        preds = torch.stack(preds, dim=1)  # (batch, T, input_dim)
        info = {
            'alpha': torch.stack(alphas, dim=1),  # (batch, T, num_experts)
            'w_norm': torch.stack(w_norms, dim=1),  # (batch, T)
            'final_w': state['w'],
            'final_m': state['m'],
        }
        return preds, info
