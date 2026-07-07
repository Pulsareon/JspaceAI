"""
改进版核心架构 v2

四大改进：
    1. 异构专家：不同模态用不同架构
    2. workspace 扩容（256）+ LayerNorm 防衰减
    3. RK4 积分（比 Euler 稳定）
    4. 更大容量
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class JSpaceConfigV2:
    input_dim: int = 32
    workspace_dim: int = 256
    expert_dim: int = 64
    num_experts: int = 12
    num_wells: int = 8
    ode_steps: int = 4
    dt: float = 0.05
    tau_w: float = 1.0
    jacobian_sparsity: int = 32
    noise_std: float = 0.005
    use_rk4: bool = True
    use_layer_norm: bool = True


class HeterogeneousExpert(nn.Module):
    """异构专家基类——模态特定编码器 + ODE 动力学"""
    def __init__(self, expert_dim, workspace_dim, input_dim, num_wells, sparsity,
                 use_rk4=True, use_layer_norm=True):
        super().__init__()
        self.expert_dim = expert_dim
        self.workspace_dim = workspace_dim
        self.use_rk4 = use_rk4

        self.encoder = self._build_encoder(input_dim, expert_dim)
        self.well_a = nn.Parameter(torch.randn(num_wells, expert_dim) * 0.2)
        self.well_b = nn.Parameter(torch.zeros(num_wells))
        self.P_in = nn.Linear(expert_dim, expert_dim, bias=False)
        self.P_out = nn.Linear(expert_dim, workspace_dim, bias=False)
        self.J_raw = nn.Parameter(torch.randn(expert_dim, workspace_dim) * 0.05)
        self.sparsity = sparsity
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.w_norm = nn.LayerNorm(expert_dim)
            self.m_norm = nn.LayerNorm(expert_dim)

    def _build_encoder(self, input_dim, expert_dim):
        raise NotImplementedError

    def get_sparse_J(self):
        if self.sparsity >= self.workspace_dim:
            return self.J_raw
        abs_J = self.J_raw.abs()
        _, topk_idx = abs_J.topk(self.sparsity, dim=-1)
        mask = torch.zeros_like(self.J_raw)
        mask.scatter_(-1, topk_idx, 1.0)
        return self.J_raw * mask

    def grad_potential(self, m):
        am = F.linear(m, self.well_a, self.well_b)
        sig = torch.sigmoid(am)
        grad_wells = torch.matmul(sig, self.well_a)
        return m - 0.5 * grad_wells

    def deriv(self, m, w, x_feat):
        J = self.get_sparse_J()
        w_proj = F.linear(w, J)
        x_proj = self.P_in(x_feat)
        grad_U = self.grad_potential(m)
        return -grad_U + w_proj + x_proj

    def forward(self, m, w, x, dt, noise_std):
        x_feat = self.encoder(x)
        if self.use_layer_norm:
            x_feat = self.w_norm(x_feat)

        if self.use_rk4:
            k1 = self.deriv(m, w, x_feat)
            k2 = self.deriv(m + 0.5*dt*k1, w, x_feat)
            k3 = self.deriv(m + 0.5*dt*k2, w, x_feat)
            k4 = self.deriv(m + dt*k3, w, x_feat)
            m_next = m + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            dm = self.deriv(m, w, x_feat)
            m_next = m + dt * dm

        if noise_std > 0:
            m_next = m_next + torch.randn_like(m_next) * noise_std
        if self.use_layer_norm:
            m_next = self.m_norm(m_next)

        contribution = self.P_out(m_next)
        return m_next, contribution


class VisualExpert(HeterogeneousExpert):
    def _build_encoder(self, input_dim, expert_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, expert_dim))

class AudioExpert(HeterogeneousExpert):
    def _build_encoder(self, input_dim, expert_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, expert_dim))

class LanguageExpert(HeterogeneousExpert):
    def _build_encoder(self, input_dim, expert_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, expert_dim))

class CrossModalExpert(HeterogeneousExpert):
    def _build_encoder(self, input_dim, expert_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, expert_dim))


class JSpaceWorkspaceV2(nn.Module):
    """改进版 workspace——LayerNorm 防衰减"""
    def __init__(self, workspace_dim, input_dim, num_experts):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.query_gen = nn.Sequential(
            nn.Linear(input_dim + workspace_dim, 128),
            nn.ReLU(), nn.Linear(128, workspace_dim))
        self.ln = nn.LayerNorm(workspace_dim)

    def forward(self, w, x, contributions, dt, tau_w):
        q = self.query_gen(torch.cat([x, w], dim=-1))
        scores = (contributions * q.unsqueeze(1)).sum(dim=-1)
        alpha = F.softmax(scores, dim=-1)
        aggregated = (alpha.unsqueeze(-1) * contributions).sum(dim=1)
        dw = (-w + aggregated) / tau_w
        w_next = w + dt * dw
        w_next = self.ln(w_next)
        return w_next, alpha


class JSpaceModelV2(nn.Module):
    """改进版 JSpace 模型——异构专家 + 大 workspace + RK4"""
    def __init__(self, config: JSpaceConfigV2):
        super().__init__()
        self.config = config
        expert_types = (
            [VisualExpert]*4 + [AudioExpert]*2 +
            [LanguageExpert]*2 + [CrossModalExpert]*4
        )[:config.num_experts]
        self.experts = nn.ModuleList([
            et(expert_dim=config.expert_dim,
               workspace_dim=config.workspace_dim,
               input_dim=config.input_dim,
               num_wells=config.num_wells,
               sparsity=config.jacobian_sparsity,
               use_rk4=config.use_rk4,
               use_layer_norm=config.use_layer_norm)
            for et in expert_types
        ])
        self.expert_modality = (
            ['visual']*2 + ['screen']*2 + ['audio']*2 +
            ['text']*2 + ['mouse']*2 + ['cross']*2
        )[:config.num_experts]
        self.workspace = JSpaceWorkspaceV2(
            workspace_dim=config.workspace_dim,
            input_dim=config.input_dim,
            num_experts=config.num_experts,
        )

    def init_state(self, batch_size, device):
        return {
            'w': torch.zeros(batch_size, self.config.workspace_dim, device=device),
            'm': [torch.zeros(batch_size, self.config.expert_dim, device=device)
                  for _ in range(self.config.num_experts)],
        }

    def step(self, state, x, record_trajectory=False):
        w, ms, cfg = state['w'], state['m'], self.config
        w_traj = []
        for _ in range(cfg.ode_steps):
            contributions, new_ms = [], []
            for i, expert in enumerate(self.experts):
                m_next, contrib = expert(ms[i], w, x, cfg.dt, cfg.noise_std)
                new_ms.append(m_next)
                contributions.append(contrib)
            contributions = torch.stack(contributions, dim=1)
            w, alpha = self.workspace(w, x, contributions, cfg.dt, cfg.tau_w)
            ms = new_ms
            if record_trajectory:
                w_traj.append(w.detach())
        return {'w': w, 'm': ms}, w_traj

    def forward(self, xs, state=None, record_trajectory=False):
        B, T = xs.shape[0], xs.shape[1]
        device = xs.device
        if state is None:
            state = self.init_state(B, device)
        outputs, w_norms = [], []
        for t in range(T):
            state, _ = self.step(state, xs[:, t], record_trajectory)
            outputs.append(state['w'])
            w_norms.append(state['w'].norm(dim=-1))
        return torch.stack(outputs, dim=1), {
            'w_norm': torch.stack(w_norms, dim=1),
        }
