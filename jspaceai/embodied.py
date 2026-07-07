"""
输出执行器层 + 神经系统

对应人类神经系统的各部分：
    - 大脑皮层: workspace w + 专家池（已在 multimodal.py）
    - 小脑: 运动控制器（前向模型+逆模型，精细动作）
    - 中枢神经: 动作调度器（反射弧+决策门控）
    - 海马体: 外部情景记忆库
    - 基底神经节: 动作价值学习（习惯化）
    - 执行器: 鼠标控制 + 键盘输出 + 音频输出 + 屏幕绘制

核心思想：输出和输入对称。
    输入：摄像头/麦克风/屏幕/键盘/鼠标 → 编码 → workspace
    输出：workspace → 解码 → 鼠标移动/键盘按键/音频播放/屏幕绘制

workspace 是模态无关的"意图空间"。
"想点击左上角"这个意图，在 workspace 里是一个向量，
解码到鼠标控制器就是移动+点击，解码到键盘就是 Tab+Enter。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import deque
from pynput import mouse as pynput_mouse
from pynput import keyboard as pynput_keyboard


# ============================================================
# 执行器层（对应手脚口）
# ============================================================

class MouseActuator:
    """
    鼠标执行器——对应"手"。

    workspace 解码出动作向量 → 移动鼠标 + 点击。
    动作空间：
        (dx, dy, click_left, click_right, scroll)
        dx, dy: 相对移动量（-1 到 1，乘以灵敏度）
        click_left/right: 0 或 1
        scroll: 滚动量
    """

    def __init__(self, sensitivity: int = 200, enabled: bool = True):
        self.sensitivity = sensitivity
        self.enabled = enabled
        self.controller = pynput_mouse.Controller() if enabled else None

    def execute(self, action: np.ndarray):
        """执行鼠标动作

        Args:
            action: (5,) = (dx, dy, click_l, click_r, scroll)
        """
        if not self.enabled or self.controller is None:
            return

        dx, dy, click_l, click_r, scroll = action

        # 移动
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            self.controller.move(int(dx * self.sensitivity), int(dy * self.sensitivity))

        # 点击
        if click_l > 0.5:
            self.controller.click(pynput_mouse.Button.left)
            time.sleep(0.05)
        if click_r > 0.5:
            self.controller.click(pynput_mouse.Button.right)
            time.sleep(0.05)

        # 滚动
        if abs(scroll) > 0.1:
            self.controller.scroll(0, int(scroll * 5))

    def get_position(self) -> tuple[int, int]:
        if self.controller:
            return self.controller.position
        return (0, 0)


class KeyboardActuator:
    """
    键盘执行器——对应"手"+"口"（打字）。

    workspace 解码出 token → 按键输入。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.controller = pynput_keyboard.Controller() if enabled else None

    def type_text(self, text: str):
        """输入文本"""
        if not self.enabled or self.controller is None:
            return
        self.controller.type(text)

    def press_key(self, key: str):
        """按下单个键"""
        if not self.enabled or self.controller is None:
            return
        try:
            self.controller.press(key)
            self.controller.release(key)
        except Exception:
            pass


class AudioActuator:
    """
    音频执行器——对应"口"（说话）。

    workspace 解码出音频波形 → 扬声器播放。
    """

    def __init__(self, sample_rate: int = 16000, enabled: bool = True):
        self.sample_rate = sample_rate
        self.enabled = enabled
        import sounddevice as sd
        self._sd = sd

    def play(self, audio: np.ndarray, blocking: bool = False):
        """播放音频"""
        if not self.enabled:
            return
        audio = np.clip(audio, -1, 1).astype(np.float32)
        self._sd.play(audio, self.sample_rate)
        if blocking:
            self._sd.wait()

    def stop(self):
        if self.enabled:
            self._sd.stop()


class ScreenActuator:
    """
    屏幕执行器——对应"手"（绘制）。

    在屏幕上绘制 workspace 解码出的图像。
    用 OpenCV 显示一个窗口。
    """

    def __init__(self, enabled: bool = True, window_name: str = "JspaceAI Output"):
        self.enabled = enabled
        self.window_name = window_name
        import cv2
        self._cv2 = cv2

    def show_image(self, img: np.ndarray):
        """显示图像"""
        if not self.enabled:
            return
        # img: (H, W, 3) RGB 或 (H, W) 灰度
        if img.ndim == 3 and img.shape[2] == 3:
            img_bgr = self._cv2.cvtColor(img.astype(np.uint8), self._cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img.astype(np.uint8)
        self._cv2.imshow(self.window_name, img_bgr)
        self._cv2.waitKey(1)

    def close(self):
        if self.enabled:
            self._cv2.destroyWindow(self.window_name)


# ============================================================
# 小脑：运动控制器（前向模型 + 逆模型）
# ============================================================

class Cerebellum(nn.Module):
    """
    小脑——运动协调与精细控制。

    前向模型：预测"如果执行动作 A，鼠标会到哪里"
    逆模型：给定"目标位置"，计算"需要什么动作"

    人类小脑学习动作的精细映射，让动作平滑准确。
    我们这里学习 workspace 意图 → 精确动作参数的映射。

    逆模型：workspace → 动作参数
    前向模型：动作参数 → 预测结果（用于误差反馈学习）
    """

    def __init__(self, workspace_dim: int, action_dim: int = 5):
        super().__init__()
        # 逆模型：workspace → action
        self.inverse_model = nn.Sequential(
            nn.Linear(workspace_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),  # 动作在 [-1, 1]
        )
        # 前向模型：action + 当前状态 → 预测下一状态
        self.forward_model = nn.Sequential(
            nn.Linear(action_dim + workspace_dim, 64),
            nn.ReLU(),
            nn.Linear(64, workspace_dim),
        )
        self.action_dim = action_dim

    def compute_action(self, w: torch.Tensor) -> torch.Tensor:
        """逆模型：从 workspace 意图计算动作"""
        return self.inverse_model(w)

    def predict_next(self, w: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向模型：预测执行动作后的 workspace 状态"""
        return self.forward_model(torch.cat([action, w], dim=-1))

    def compute_motor_error(self, w: torch.Tensor, action: torch.Tensor,
                            w_actual_next: torch.Tensor) -> torch.Tensor:
        """计算运动误差——用于小脑学习"""
        w_pred = self.predict_next(w, action)
        return F.mse_loss(w_pred, w_actual_next)


# ============================================================
# 中枢神经：动作调度器
# ============================================================

@dataclass
class ReflexArc:
    """反射弧——不经过大脑的快速反应"""
    trigger: str          # 触发条件描述
    condition: callable   # 检查函数
    action: callable      # 执行函数
    priority: int = 0     # 优先级


class CentralNervousSystem:
    """
    中枢神经系统——动作调度。

    功能：
        1. 反射弧：快速反应，不经过 workspace
           - 如：突然大声音 → 退缩
           - 如：屏幕突然变暗 → 警觉
        2. 决策门控：决定是否让 workspace 的意图执行
           - 高风险动作需要"确认"
           - 习惯化动作直接执行
        3. 动作序列：把复杂意图拆成动作序列
           - 如"点击按钮"→ 移动到位置 + 点击

    这是"自由意志"的工程对应——不是所有意图都执行，
    系统有一个门控机制决定哪些意图变成行动。
    """

    def __init__(self):
        self.reflexes: list[ReflexArc] = []
        self.action_history: deque = deque(maxlen=100)
        self.inhibit_score: float = 0.0  # 抑制分数，高时阻止动作

    def add_reflex(self, reflex: ReflexArc):
        self.reflexes.append(reflex)
        self.reflexes.sort(key=lambda r: -r.priority)

    def check_reflexes(self, sensory_state: dict) -> Optional[callable]:
        """检查是否有反射触发"""
        for reflex in self.reflexes:
            try:
                if reflex.condition(sensory_state):
                    return reflex.action
            except Exception:
                continue
        return None

    def should_execute(self, action_strength: float, risk: float = 0.0) -> bool:
        """决策门控：是否执行动作

        Args:
            action_strength: 动作强度（workspace 驱动）
            risk: 风险评估（0-1）

        Returns:
            是否执行
        """
        # 抑制分数高时不执行
        threshold = 0.3 + risk * 0.5 + self.inhibit_score
        return action_strength > threshold

    def record_action(self, action: np.ndarray, modality: str):
        """记录执行的动作"""
        self.action_history.append({
            'time': time.time(),
            'action': action.tolist() if hasattr(action, 'tolist') else action,
            'modality': modality,
        })


# ============================================================
# 海马体：外部情景记忆库
# ============================================================

class Hippocampus:
    """
    海马体——情景记忆。

    存储历史 workspace 快照 + 时间戳 + 上下文。
    当前 workspace 可以"回忆"相似的历史状态。

    人类的情景记忆："我记得昨天在那个房间里说了什么"
    对应：检索与当前 workspace 相似的历史 workspace。

    实现用简单的向量数据库（numpy + cosine similarity）。
    """

    def __init__(self, capacity: int = 1000, workspace_dim: int = 64):
        self.capacity = capacity
        self.workspace_dim = workspace_dim
        self.memories: deque = deque(maxlen=capacity)

    def store(self, w: np.ndarray, context: dict = None):
        """存储一个 workspace 快照"""
        self.memories.append({
            'w': w.copy(),
            'context': context or {},
            'timestamp': time.time(),
        })

    def recall(self, w_query: np.ndarray, top_k: int = 3) -> list[dict]:
        """检索相似的历史记忆

        Args:
            w_query: 当前 workspace
            top_k: 返回最相似的 k 个

        Returns:
            list of {w, context, timestamp, similarity}
        """
        if not self.memories:
            return []

        # 计算相似度
        similarities = []
        for mem in self.memories:
            sim = np.dot(w_query, mem['w']) / (
                np.linalg.norm(w_query) * np.linalg.norm(mem['w']) + 1e-8
            )
            similarities.append(sim)

        # 取 top-k
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_idx:
            mem = self.memories[idx]
            results.append({
                'w': mem['w'],
                'context': mem['context'],
                'timestamp': mem['timestamp'],
                'similarity': similarities[idx],
            })
        return results

    def size(self) -> int:
        return len(self.memories)


# ============================================================
# 基底神经节：动作价值学习
# ============================================================

class BasalGanglia:
    """
    基底神经节——习惯学习与动作选择。

    学习"在什么状态下执行什么动作价值多少"。
    高频执行的（workspace, action）对会"习惯化"——直接执行不经过思考。

    对应人类的习惯：开车的动作熟练后不需要思考，
    就是基底神经节接管了动作选择。
    """

    def __init__(self, workspace_dim: int = 64, n_actions: int = 5,
                 learning_rate: float = 0.01):
        self.workspace_dim = workspace_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        # Q-table 的近似：用线性函数 Q(s, a) = w_a · s
        self.action_weights = np.zeros((n_actions, workspace_dim))
        # 习惯化计数
        self.habit_counts = np.zeros(n_actions)

    def compute_values(self, w: np.ndarray) -> np.ndarray:
        """计算各动作的价值 Q(s, a)"""
        return self.action_weights @ w  # (n_actions,)

    def select_action(self, w: np.ndarray, exploration: float = 0.1) -> int:
        """选择动作（ε-greedy）"""
        values = self.compute_values(w)
        if np.random.random() < exploration:
            return np.random.randint(self.n_actions)
        return np.argmax(values)

    def update(self, w: np.ndarray, action: int, reward: float):
        """更新动作价值（TD learning 简化版）"""
        values = self.compute_values(w)
        td_error = reward - values[action]
        self.action_weights[action] += self.lr * td_error * w
        self.habit_counts[action] += 1

    def is_habitual(self, action: int, threshold: int = 10) -> bool:
        """判断动作是否已习惯化"""
        return self.habit_counts[action] >= threshold


# ============================================================
# 完整的具身 Agent
# ============================================================

class EmbodiedAgent:
    """
    完整的具身 Agent——感知-思考-行动闭环。

    结构（对应人类神经系统）：
        感知层（眼耳皮肤）→ FullSensoryStream
            ↓ 编码
        大脑皮层（思考）→ MultimodalJSpaceModel
            ↓ workspace w
        海马体（记忆）→ Hippocampus 存储和检索
            ↓
        基底神经节（动作选择）→ BasalGanglia 选动作
            ↓
        小脑（运动控制）→ Cerebellum 精细化动作
            ↓
        中枢神经（门控）→ CentralNervousSystem 决定是否执行
            ↓
        执行器（手足口）→ MouseActuator + KeyboardActuator + AudioActuator

    循环：
        1. 感知：从五感获取输入
        2. 思考：workspace 更新
        3. 回忆：海马体检索相关记忆
        4. 决策：基底神经节选动作
        5. 精化：小脑计算动作参数
        6. 门控：中枢神经决定执行
        7. 行动：执行器执行
        8. 学习：更新基底神经节、小脑、海马体
    """

    def __init__(self, model, device: str = 'cpu',
                 enable_mouse_output: bool = False,
                 enable_keyboard_output: bool = False,
                 enable_audio_output: bool = True,
                 enable_screen_output: bool = True,
                 enable_memory: bool = True,
                 risk_threshold: float = 0.3):
        self.model = model
        self.device = device
        self.config = model.config

        # 感知层
        from .desktop import FullSensoryStream
        from .platform import get_screen_size
        self.screen_w, self.screen_h = get_screen_size()
        self.senses = FullSensoryStream(
            use_camera=True, use_mic=True, use_desktop=True,
            img_size=(self.config.img_size, self.config.img_size),
        )

        # 执行器层
        self.mouse_actuator = MouseActuator(enabled=enable_mouse_output)
        self.keyboard_actuator = KeyboardActuator(enabled=enable_keyboard_output)
        self.audio_actuator = AudioActuator(enabled=enable_audio_output)
        self.screen_actuator = ScreenActuator(enabled=enable_screen_output)

        # 神经系统
        self.cerebellum = Cerebellum(
            workspace_dim=self.config.workspace_dim,
            action_dim=5,  # (dx, dy, click_l, click_r, scroll)
        ).to(device)
        self.cns = CentralNervousSystem()
        self.hippocampus = Hippocampus(
            workspace_dim=self.config.workspace_dim,
        ) if enable_memory else None
        self.basal_ganglia = BasalGanglia(
            workspace_dim=self.config.workspace_dim,
            n_actions=5,
        )

        # 内部状态
        self.state = model.init_state(1, torch.device(device))
        self.step_count = 0
        self.running = False
        self.risk_threshold = risk_threshold

        # 设置反射弧
        self._setup_reflexes()

    def _setup_reflexes(self):
        """设置基本反射"""
        # 反射1：大声音 → 抑制动作
        def loud_noise_condition(state):
            audio = state.get('audio')
            if audio and hasattr(audio, 'data'):
                return np.abs(audio.data).mean() > 0.5
            return False

        def loud_noise_response():
            self.cns.inhibit_score = 0.5

        self.cns.add_reflex(ReflexArc(
            trigger='loud_noise',
            condition=loud_noise_condition,
            action=loud_noise_response,
            priority=10,
        ))

    def perceive(self) -> dict:
        """感知：从五感获取输入"""
        return self.senses.get_latest()

    def think(self, sensory_data: dict) -> tuple[torch.Tensor, str]:
        """思考：更新 workspace

        Returns:
            w: workspace 状态
            modality: 本次输入的模态
        """
        # 按优先级选模态
        modality = None
        input_tensor = None

        if sensory_data['keyboard']:
            key_ids = [ord(ev.data[0]) if ev.data and len(ev.data) == 1 else 0
                       for ev in sensory_data['keyboard'][:8]]
            if key_ids:
                modality = 'keyboard'
                input_tensor = torch.tensor([key_ids], dtype=torch.long).to(self.device)

        elif sensory_data['mouse']:
            ev = sensory_data['mouse'][0]
            x, y = ev.data
            modality = 'mouse'
            input_tensor = torch.tensor([[
                x / self.screen_w, y / self.screen_h,
                1.0 if ev.modifiers.get('button') == 'left' else 0.0,
                1.0 if ev.modifiers.get('button') == 'right' else 0.0,
            ]], dtype=torch.float32).to(self.device)

        elif sensory_data['audio']:
            modality = 'audio'
            input_tensor = torch.tensor([sensory_data['audio'].data],
                                       dtype=torch.float32).to(self.device)

        elif sensory_data['screen']:
            img = sensory_data['screen'].data
            modality = 'screen'
            input_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1.0

        elif sensory_data['camera']:
            img = sensory_data['camera'].data
            modality = 'image'
            input_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1.0

        if modality is None or input_tensor is None:
            return self.state['w'], 'idle'

        # 编码 + forward
        with torch.no_grad():
            x = self.model.encode_modality(modality, input_tensor)
            if x.dim() == 1: x = x.unsqueeze(0)
            if x.dim() == 3: x = x[:, -1, :]
            if x.shape[0] != 1: x = x[-1:]
            self.state, _ = self.model.step(self.state, x)

        return self.state['w'], modality

    def remember(self, w: torch.Tensor, context: dict):
        """存储到海马体"""
        if self.hippocampus:
            self.hippocampus.store(w[0].cpu().numpy(), context)

    def recall_memories(self, w: torch.Tensor, top_k: int = 3) -> list:
        """从海马体回忆"""
        if self.hippocampus:
            return self.hippocampus.recall(w[0].cpu().numpy(), top_k)
        return []

    def decide_and_act(self, w: torch.Tensor, modality: str) -> dict:
        """决策和行动

        1. 基底神经节选动作
        2. 小脑计算动作参数
        3. 中枢神经门控
        4. 执行器执行
        """
        w_np = w[0].cpu().numpy()

        # 1. 基底神经节：选动作类别
        action_idx = self.basal_ganglia.select_action(w_np, exploration=0.2)

        # 2. 小脑：计算精确动作参数
        with torch.no_grad():
            action_params = self.cerebellum.compute_action(w)[0].cpu().numpy()

        # 3. 中枢神经：门控
        action_strength = np.abs(action_params).max()
        risk = 0.0
        # 鼠标点击风险较高
        if action_params[2] > 0.5 or action_params[3] > 0.5:
            risk = 0.5

        execute = self.cns.should_execute(action_strength, risk)

        # 4. 执行
        action_taken = None
        if execute:
            # 鼠标动作
            self.mouse_actuator.execute(action_params)
            # 音频输出（从 workspace 解码）
            if self.step_count % 10 == 0:
                with torch.no_grad():
                    audio_out = self.model.audio_decoder(w)[0].cpu().numpy()
                self.audio_actuator.play(audio_out)
            # 屏幕输出
            if self.screen_actuator.enabled:
                with torch.no_grad():
                    img_out = self.model.visual_decoder(w)[0].cpu().permute(1, 2, 0).numpy()
                    img_out = ((img_out + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    self.screen_actuator.show_image(img_out)

            action_taken = action_params.tolist()
            self.cns.record_action(action_params, modality)

        return {
            'action_idx': action_idx,
            'action_params': action_params.tolist(),
            'executed': execute,
            'action_strength': float(action_strength),
            'risk': float(risk),
        }

    def learn(self, w: torch.Tensor, action_params: np.ndarray, reward: float = 0.0):
        """学习——更新基底神经节和小脑"""
        w_np = w[0].cpu().numpy()

        # 基底神经节：更新动作价值
        action_idx = self.basal_ganglia.select_action(w_np, exploration=0.0)
        self.basal_ganglia.update(w_np, action_idx, reward)

    def step_once(self) -> dict:
        """执行一步完整的感知-思考-行动循环"""
        # 1. 感知
        sensory_data = self.perceive()

        # 2. 检查反射
        reflex_action = self.cns.check_reflexes(sensory_data)
        if reflex_action:
            reflex_action()

        # 3. 思考
        w, modality = self.think(sensory_data)

        # 4. 回忆
        memories = self.recall_memories(w)

        # 5. 决策和行动
        action_info = self.decide_and_act(w, modality)

        # 6. 学习（自监督：预测误差作为 reward）
        reward = -action_info['risk']  # 简化：风险越低 reward 越高
        self.learn(w, np.array(action_info['action_params']), reward)

        # 7. 记忆存储
        self.remember(w, {
            'modality': modality,
            'action': action_info,
            'step': self.step_count,
        })

        self.step_count += 1

        return {
            'step': self.step_count,
            'modality': modality,
            'w_norm': w.norm().item(),
            'action': action_info,
            'memories_count': self.hippocampus.size() if self.hippocampus else 0,
        }

    def run(self, n_steps: int = 100, interval: float = 0.2,
            on_step: callable = None):
        """运行感知-思考-行动循环"""
        self.running = True
        self.senses.start()
        print(f"具身 Agent 启动，{n_steps} 步")
        print(f"执行器: mouse={self.mouse_actuator.enabled}, "
              f"keyboard={self.keyboard_actuator.enabled}, "
              f"audio={self.audio_actuator.enabled}, "
              f"screen={self.screen_actuator.enabled}")
        print("=" * 60)

        step_log = []
        try:
            for _ in range(n_steps):
                if not self.running:
                    break
                info = self.step_once()
                step_log.append(info)

                if on_step:
                    on_step(info)
                elif info['step'] % 5 == 0:
                    print(f"  step {info['step']:3d} | mod {info['modality']:8s} | "
                          f"||w|| {info['w_norm']:.3f} | "
                          f"action {info['action']['action_idx']} | "
                          f"executed {info['action']['executed']}")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.running = False
            self.senses.stop()
            self.audio_actuator.stop()
            if self.screen_actuator.enabled:
                self.screen_actuator.close()

        return step_log
