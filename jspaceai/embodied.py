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
import numpy as np
import time
from pynput import mouse as pynput_mouse
from pynput import keyboard as pynput_keyboard

from .consensus import ConsensusSnapshot
from .events import WorkspaceEvent
from .memory import Hippocampus, InMemoryVectorMemoryStore
from .policy import (
    ActionPolicy,
    BasalGanglia,
    Cerebellum,
    CentralNervousSystem,
    ReflexArc,
    compose_action_params,
)

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
# 完整的具身 Agent
# ============================================================

class EmbodiedAgent:
    """
    完整的具身 Agent——感知-思考-行动闭环。

    结构：
        感知层（眼耳皮肤）→ FullSensoryStream
            ↓ 编码
        大脑皮层（思考）→ MultimodalJSpaceModel
            ↓ workspace w
        海马体（记忆）→ Hippocampus 存储和检索
            ↓
        ActionPolicy（价值 + 电机参数 + 门控）
            ↓
        执行器（手足口）→ MouseActuator + KeyboardActuator + AudioActuator

    循环：
        1. 感知：从五感获取输入
        2. 思考：workspace 更新
        3. 回忆：海马体检索相关记忆
        4. 决策：ActionPolicy 选动作并门控
        7. 行动：执行器执行
        8. 学习：更新策略与海马体
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

        # 策略层
        self.policy = ActionPolicy(
            workspace_dim=self.config.workspace_dim,
            action_dim=5,
            n_actions=5,
            base_threshold=risk_threshold,
            device=device,
        )
        self.cerebellum = self.policy.motor_controller
        self.cns = self.policy.gate
        self.hippocampus = InMemoryVectorMemoryStore(
            workspace_dim=self.config.workspace_dim,
        ) if enable_memory else None
        self.basal_ganglia = self.policy.value_model

        # 内部状态
        self.state = model.init_state(1, torch.device(device))
        self.step_count = 0
        self.running = False
        self.risk_threshold = risk_threshold
        self.last_consensus: ConsensusSnapshot | None = None

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

        self.policy.add_reflex(ReflexArc(
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
            self.state, alpha, _ = self.model.step(self.state, x)
            self.last_consensus = ConsensusSnapshot.from_workspace(
                self.state['w'], alpha, modality, self.model.expert_modality
            )

        return self.state['w'], modality

    def remember(self, w: torch.Tensor, context: dict):
        """存储 workspace event 到记忆层"""
        if self.hippocampus:
            event = WorkspaceEvent.from_tensor(
                w,
                modality=str(context.get('modality', 'unknown')),
                step=int(context.get('step', self.step_count)),
                consensus=context.get('consensus', self.last_consensus),
                context=context,
            )
            self.hippocampus.put(event)

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
        decision = self.policy.decide(w)
        action_params = decision.action_params
        execute = decision.should_execute

        # 4. 执行
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

            self.policy.record_action(action_params, modality)

        return decision.to_dict(executed=execute)

    def learn(self, w: torch.Tensor, action_idx: int, reward: float = 0.0):
        """学习——更新基底神经节和小脑"""
        self.policy.learn(w, action_idx, reward)

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
        self.recall_memories(w)

        # 5. 决策和行动
        action_info = self.decide_and_act(w, modality)

        # 6. 学习（自监督：预测误差作为 reward）
        reward = -action_info['risk']  # 简化：风险越低 reward 越高
        self.learn(w, action_info['action_idx'], reward)

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
            'consensus': self.last_consensus.to_dict() if self.last_consensus else None,
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
                          f"action {info['action']['action_name']} | "
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
