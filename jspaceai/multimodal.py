"""
多模态感知-行动系统

原生支持图片、音频、视频、文本四种模态。
所有模态编码到统一的 workspace 向量空间，从 workspace 解码到任意模态。

设计原则：
    1. 原生多模态——不经过文本中间表示
    2. 流式处理——支持实时麦克风/摄像头输入
    3. 闭环——输出反馈到输入
    4. 在线学习——持续进化

模态专家分工（对应 Anthropic 论文的"专家分工"）：
    - 视觉专家：处理图片/视频帧
    - 听觉专家：处理音频频谱
    - 语言专家：处理文本 token
    - 跨模态专家：对齐不同模态的概念

每个专家有自己的内部状态 m_i，通过 J-space 在 workspace w 中广播。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .core import JSpaceConfig, Expert, JSpaceWorkspace


@dataclass
class MultimodalConfig(JSpaceConfig):
    """多模态配置"""
    # 文本
    vocab_size: int = 100
    embed_dim: int = 16
    # 视觉
    img_channels: int = 3
    img_size: int = 32           # 编码后的图像尺寸（原始会 resize）
    visual_feature_dim: int = 32
    # 音频
    audio_sample_rate: int = 16000
    audio_frame_size: int = 1024  # 每帧采样数
    audio_feature_dim: int = 32
    # 共享
    workspace_dim: int = 64       # 比 unimodal 大，承载多模态
    num_experts: int = 12         # 12 个专家：2视觉 + 2屏幕 + 2听觉 + 2语言 + 2鼠标 + 2跨模态
    expert_dim: int = 24
    # 键盘/鼠标
    keyboard_vocab: int = 128     # 键盘字符词汇表
    mouse_feature_dim: int = 16   # 鼠标特征维度


class VisualEncoder(nn.Module):
    """
    视觉编码器：图片 → workspace 向量

    轻量 CNN，不用预训练。从零学。
    输入：(batch, 3, H, W)
    输出：(batch, input_dim)  投影到 workspace 输入空间
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16→8
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 8→4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局池化 → (32, 1, 1)
        )
        self.proj = nn.Linear(32, input_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (batch, 3, H, W) 或 (batch, T, 3, H, W) 视频序列
        returns: (batch, input_dim) 或 (batch, T, input_dim)
        """
        if img.dim() == 5:  # 视频序列
            B, T, C, H, W = img.shape
            img = img.reshape(B * T, C, H, W)
            feat = self.conv(img).squeeze(-1).squeeze(-1)  # (B*T, 32)
            feat = self.proj(feat)  # (B*T, input_dim)
            return feat.reshape(B, T, -1)
        else:
            feat = self.conv(img).squeeze(-1).squeeze(-1)
            return self.proj(feat)


class AudioEncoder(nn.Module):
    """
    音频编码器：音频波形 → workspace 向量

    轻量 1D CNN 处理原始波形。
    输入：(batch, audio_frame_size) 原始音频采样
    输出：(batch, input_dim)
    """

    def __init__(self, input_dim: int, audio_frame_size: int = 1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 64, stride=4, padding=32),  # 下采样
            nn.ReLU(),
            nn.Conv1d(16, 32, 32, stride=4, padding=16),
            nn.ReLU(),
            nn.Conv1d(32, 32, 16, stride=2, padding=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, input_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (batch, frame_size) 或 (batch, T, frame_size)
        returns: (batch, input_dim) 或 (batch, T, input_dim)
        """
        if audio.dim() == 3:
            B, T, F = audio.shape
            audio = audio.reshape(B * T, F)
            x = audio.unsqueeze(1)  # (B*T, 1, F)
            feat = self.conv(x).squeeze(-1)
            feat = self.proj(feat)
            return feat.reshape(B, T, -1)
        else:
            x = audio.unsqueeze(1)  # (B, 1, F)
            feat = self.conv(x).squeeze(-1)
            return self.proj(feat)


class TextEncoder(nn.Module):
    """文本编码器：token → workspace 向量"""

    def __init__(self, vocab_size: int, embed_dim: int, input_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, input_dim, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch,) 或 (batch, T)"""
        if token_ids.dim() == 1:
            emb = self.embedding(token_ids)
            return self.proj(emb)
        else:
            emb = self.embedding(token_ids)  # (B, T, embed)
            return self.proj(emb)


class KeyboardEncoder(nn.Module):
    """
    键盘编码器：按键序列 → workspace 向量

    把键盘按键序列编码成语义向量。
    和 TextEncoder 类似，但字符集不同（含特殊键）。
    """

    def __init__(self, vocab_size: int, embed_dim: int, input_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, input_dim, bias=False)

    def forward(self, key_ids: torch.Tensor) -> torch.Tensor:
        """key_ids: (batch,) 或 (batch, T)"""
        if key_ids.dim() == 1:
            return self.proj(self.embedding(key_ids))
        else:
            return self.proj(self.embedding(key_ids))


class MouseEncoder(nn.Module):
    """
    鼠标编码器：鼠标位置 + 事件 → workspace 向量

    输入：(batch, 4) = (x_norm, y_norm, click_left, click_right)
        x_norm, y_norm: 归一化到 [0, 1] 的鼠标坐标
        click_left, click_right: 0 或 1
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, mouse_data: torch.Tensor) -> torch.Tensor:
        """mouse_data: (batch, 4) → (batch, input_dim)"""
        return self.net(mouse_data)


class VisualDecoder(nn.Module):
    """视觉解码器：workspace 向量 → 图片"""

    def __init__(self, workspace_dim: int, img_size: int = 32):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(workspace_dim, 32 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 16→32
            nn.Tanh(),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (batch, workspace_dim) → (batch, 3, 32, 32)"""
        x = self.fc(w).reshape(-1, 32, 4, 4)
        return self.deconv(x)


class AudioDecoder(nn.Module):
    """音频解码器：workspace 向量 → 音频波形"""

    def __init__(self, workspace_dim: int, audio_frame_size: int = 1024):
        super().__init__()
        self.audio_frame_size = audio_frame_size
        self.fc = nn.Linear(workspace_dim, 32 * 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 32, stride=4, padding=14, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 16, stride=4, padding=6, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 64, stride=4, padding=30, output_padding=2),
            nn.Tanh(),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (batch, workspace_dim) → (batch, audio_frame_size)"""
        x = self.fc(w).reshape(-1, 32, 64)
        audio = self.deconv(x).squeeze(1)  # (batch, L)
        # 裁剪或填充到目标长度
        if audio.shape[-1] > self.audio_frame_size:
            audio = audio[:, :self.audio_frame_size]
        else:
            audio = F.pad(audio, (0, self.audio_frame_size - audio.shape[-1]))
        return audio


class TextDecoder(nn.Module):
    """文本解码器：workspace 向量 → logits"""

    def __init__(self, workspace_dim: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim, 64),
            nn.Tanh(),
            nn.Linear(64, vocab_size),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.net(w)


class MultimodalJSpaceModel(nn.Module):
    """
    多模态 JSpace 模型。

    12 个专家分工：
        专家 0,1:   视觉（处理摄像头图片）
        专家 2,3:   屏幕（处理屏幕截图）
        专家 4,5:   听觉（处理音频帧）
        专家 6,7:   语言（处理文本 token）
        专家 8,9:   鼠标（处理鼠标位置/点击）
        专家 10,11: 跨模态（对齐不同模态的概念）

    所有专家共享 workspace w，通过 J-space 广播。

    支持的输入模态：
        'image'    - 摄像头图片 (batch, 3, H, W)
        'screen'   - 屏幕截图 (batch, 3, H, W)（用视觉编码器）
        'audio'    - 音频帧 (batch, frame_size)
        'text'     - 文本 token (batch,)
        'keyboard' - 键盘按键 (batch,) 或 (batch, T)
        'mouse'    - 鼠标数据 (batch, 4) = (x, y, click_l, click_r)
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # 编码器（各模态 → input_dim）
        self.visual_encoder = VisualEncoder(config.input_dim)        # 摄像头 + 屏幕
        self.audio_encoder = AudioEncoder(config.input_dim, config.audio_frame_size)
        self.text_encoder = TextEncoder(config.vocab_size, config.embed_dim, config.input_dim)
        self.keyboard_encoder = KeyboardEncoder(config.keyboard_vocab, config.embed_dim, config.input_dim)
        self.mouse_encoder = MouseEncoder(config.input_dim)

        # 解码器（workspace → 各模态）
        self.visual_decoder = VisualDecoder(config.workspace_dim, config.img_size)
        self.audio_decoder = AudioDecoder(config.workspace_dim, config.audio_frame_size)
        self.text_decoder = TextDecoder(config.workspace_dim, config.vocab_size)

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

        # 模态类型标记（12 个专家的分工）
        self.expert_modality = (
            ['visual'] * 2 +      # 0,1: 摄像头
            ['screen'] * 2 +      # 2,3: 屏幕
            ['audio'] * 2 +       # 4,5: 听觉
            ['text'] * 2 +        # 6,7: 语言
            ['mouse'] * 2 +       # 8,9: 鼠标
            ['cross'] * 2         # 10,11: 跨模态
        )[:config.num_experts]

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            'w': torch.zeros(batch_size, self.config.workspace_dim, device=device),
            'm': [torch.zeros(batch_size, self.config.expert_dim, device=device)
                  for _ in range(self.config.num_experts)],
        }

    def step(self, state: dict, x: torch.Tensor,
             record_trajectory: bool = False) -> tuple[dict, list]:
        """
        单步前向

        Args:
            state: {'w': ..., 'm': [...]}
            x: (batch, input_dim) 已编码的输入（任意模态）
            record_trajectory: 是否记录 w 轨迹

        Returns:
            new_state, w_trajectory (list)
        """
        w = state['w']
        ms = state['m']
        cfg = self.config
        w_trajectory = []

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

        new_state = {'w': w, 'm': ms}
        return new_state, w_trajectory

    def encode_modality(self, modality: str, data: torch.Tensor) -> torch.Tensor:
        """编码任意模态到 input_dim

        支持的模态：image, screen, audio, text, keyboard, mouse
        - image/screen: 共用 visual_encoder（都是 RGB 图像）
        - keyboard: 键盘按键 id
        - mouse: (batch, 4) = (x_norm, y_norm, click_left, click_right)
        """
        if modality in ('image', 'screen'):
            return self.visual_encoder(data)
        elif modality == 'audio':
            return self.audio_encoder(data)
        elif modality == 'text':
            return self.text_encoder(data)
        elif modality == 'keyboard':
            return self.keyboard_encoder(data)
        elif modality == 'mouse':
            return self.mouse_encoder(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def decode_modality(self, modality: str, w: torch.Tensor) -> torch.Tensor:
        """从 workspace 解码到任意模态

        screen 用 visual_decoder（和 image 共享）
        keyboard 用 text_decoder（都是 token）
        """
        if modality in ('image', 'screen'):
            return self.visual_decoder(w)
        elif modality == 'audio':
            return self.audio_decoder(w)
        elif modality in ('text', 'keyboard'):
            return self.text_decoder(w)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def forward_multimodal(self, modality: str, data: torch.Tensor,
                           state: dict | None = None,
                           record_trajectory: bool = False) -> tuple[dict, dict]:
        """
        多模态前向

        Args:
            modality: 'image' / 'audio' / 'text'
            data: 模态原始数据
            state: 初始状态
            record_trajectory: 是否记录 w 轨迹

        Returns:
            outputs: {'w': workspace, 'logits/img/audio': 各模态解码}
            info: {'alpha', 'w_norm', 'w_trajectory'}
        """
        if state is None:
            # 先编码确定 batch size
            x = self.encode_modality(modality, data)
            batch_size = x.shape[0]
            device = x.device
            state = self.init_state(batch_size, device)
        else:
            x = self.encode_modality(modality, data)

        # 处理序列或单步
        if x.dim() == 2:  # 单步 (batch, input_dim)
            state, w_traj = self.step(state, x, record_trajectory)
            w = state['w']
        else:  # 序列 (batch, T, input_dim)
            w_traj_all = []
            for t in range(x.shape[1]):
                state, w_traj = self.step(state, x[:, t], record_trajectory)
                if record_trajectory:
                    w_traj_all.append(w_traj)
            w = state['w']

        # 解码到所有模态（workspace 是模态无关的，可解码到任意模态）
        outputs = {
            'w': w,
            'image': self.visual_decoder(w),
            'audio': self.audio_decoder(w),
            'text_logits': self.text_decoder(w),
        }

        info = {
            'w_norm': w.norm(dim=-1),
        }
        if record_trajectory and w_traj_all:
            info['w_trajectory'] = w_traj_all

        return outputs, info
