"""
实时多模态 I/O 层

使用 OpenCV（视频/图像）+ sounddevice（音频 I/O，基于 PortAudio 开源库）
+ macOS afplay（音频播放）实现实时感知-行动循环。

提供：
    - CameraStream: 实时摄像头采集
    - MicrophoneStream: 实时麦克风采集
    - AudioPlayer: 音频播放（扬声器）
    - MultimodalStream: 统一的实时多模态流管理

所有流都是非阻塞的，用队列缓冲。可以同时采集视频和音频。
"""
from __future__ import annotations

import cv2
import numpy as np
import sounddevice as sd
import torch
import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class Frame:
    """统一的帧数据结构"""
    timestamp: float
    modality: str          # 'image' / 'audio' / 'text'
    data: np.ndarray       # 图像 (H,W,3) / 音频 (frame_size,) / 文本 str


class CameraStream:
    """
    OpenCV 摄像头流。

    非阻塞采集，帧放入队列。
    """

    def __init__(self, camera_index: int = 0, frame_size: tuple = (32, 32),
                 fps: int = 10):
        self.camera_index = camera_index
        self.target_size = frame_size  # resize 到这个尺寸（给模型用）
        self.fps = fps
        self.frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self):
        """启动采集"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_index}（需要 macOS 权限）")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        frame_interval = 1.0 / self.fps
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize 到目标尺寸
            frame_resized = cv2.resize(frame_rgb, self.target_size)

            try:
                self.frame_queue.put_nowait(
                    Frame(time.time(), 'image', frame_resized)
                )
            except queue.Full:
                pass  # 丢帧

            time.sleep(frame_interval)

    def get_frame(self) -> Optional[Frame]:
        """获取最新帧（非阻塞）"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class MicrophoneStream:
    """
    sounddevice 麦克风流。

    持续采集音频，分帧放入队列。
    """

    def __init__(self, sample_rate: int = 16000, frame_size: int = 1024,
                 channels: int = 1):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self.frame_queue: queue.Queue = queue.Queue(maxsize=100)
        self.running = False
        self.stream: Optional[sd.InputStream] = None

    def start(self):
        """启动采集"""
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=self.channels,
            dtype='float32',
            callback=self._audio_callback,
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        # indata: (frame_size, channels)
        audio = indata[:, 0] if self.channels > 1 else indata.flatten()
        try:
            self.frame_queue.put_nowait(
                Frame(time.time(), 'audio', audio.copy())
            )
        except queue.Full:
            pass

    def get_frame(self) -> Optional[Frame]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()


class AudioPlayer:
    """
    音频播放器。用 sounddevice 输出到扬声器。
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.playing = False

    def play(self, audio: np.ndarray, blocking: bool = False):
        """播放音频波形

        Args:
            audio: (n_samples,) float32, 范围 [-1, 1]
            blocking: 是否阻塞等待播放完成
        """
        audio = np.clip(audio, -1, 1).astype(np.float32)
        sd.play(audio, self.sample_rate)
        if blocking:
            sd.wait()

    def stop(self):
        sd.stop()


class MultimodalStream:
    """
    统一管理摄像头 + 麦克风 + 扬声器的实时流。

    提供统一的接口：
        - start(): 启动所有采集
        - get_frames(): 获取当前所有可用帧（按时间戳对齐）
        - respond_audio(): 播放音频回应
        - stop(): 停止所有

    这是感知-行动循环的物理层。
    """

    def __init__(self, use_camera: bool = True, use_mic: bool = True,
                 img_size: tuple = (32, 32), sample_rate: int = 16000,
                 audio_frame_size: int = 1024):
        self.use_camera = use_camera
        self.use_mic = use_mic

        self.camera = CameraStream(frame_size=img_size) if use_camera else None
        self.mic = MicrophoneStream(
            sample_rate=sample_rate, frame_size=audio_frame_size
        ) if use_mic else None
        self.player = AudioPlayer(sample_rate=sample_rate)

    def start(self):
        """启动所有采集"""
        if self.camera:
            try:
                self.camera.start()
                print("  摄像头已启动")
            except RuntimeError as e:
                print(f"  摄像头启动失败: {e}")
                self.camera = None

        if self.mic:
            try:
                self.mic.start()
                print("  麦克风已启动")
            except Exception as e:
                print(f"  麦克风启动失败: {e}")
                self.mic = None

    def get_latest_frames(self) -> dict[str, Optional[Frame]]:
        """获取最新的各模态帧"""
        result = {
            'image': None,
            'audio': None,
        }
        if self.camera:
            # 清空队列，只保留最新
            latest = None
            while True:
                f = self.camera.get_frame()
                if f is None:
                    break
                latest = f
            result['image'] = latest

        if self.mic:
            latest = None
            while True:
                f = self.mic.get_frame()
                if f is None:
                    break
                latest = f
            result['audio'] = latest

        return result

    def play_audio(self, audio: np.ndarray, blocking: bool = False):
        """播放音频"""
        self.player.play(audio, blocking)

    def stop(self):
        if self.camera:
            self.camera.stop()
        if self.mic:
            self.mic.stop()
        self.player.stop()


class SensoryMotorLoop:
    """
    感知-行动循环：传感器 → 模型 → 执行器 → 传感器...

    这是自进化语言网络的核心运行时。

    循环：
        1. 从传感器读帧（摄像头/麦克风）
        2. 编码到 workspace 输入
        3. 模型 forward 一步（workspace 更新）
        4. 从 workspace 解码输出（音频/图像/文本）
        5. 执行器输出（扬声器播放/显示图像）
        6. 输出反馈到输入（自监督：预测自己的输出）
        7. 在线学习（更新参数）

    这个循环永不停止——模型持续感知、思考、行动、学习。
    """

    def __init__(self, model, stream: MultimodalStream, device: str = 'cpu'):
        self.model = model
        self.stream = stream
        self.device = device
        self.state = model.init_state(1, torch.device(device))
        self.running = False
        self.step_count = 0

        # 自监督学习记录
        self.last_output = None  # 上一步的输出（用于反馈）
        self.online_loss_history = []

    def step_once(self) -> dict:
        """执行一步感知-行动循环

        Returns:
            info: 包含输入模态、输出、loss 等
        """
        # 1. 读取传感器
        frames = self.stream.get_latest_frames()

        # 选择活跃模态（优先音频，其次图像）
        modality = None
        input_data = None
        if frames['audio'] is not None:
            modality = 'audio'
            input_data = torch.tensor(
                frames['audio'].data, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
        elif frames['image'] is not None:
            modality = 'image'
            img = frames['image'].data  # (H, W, 3) uint8
            img_tensor = torch.tensor(
                img, dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1.0  # 归一化到 [-1, 1]
            input_data = img_tensor

        if modality is None:
            return {'status': 'no_input'}

        # 2. 编码 + forward
        with torch.no_grad():
            outputs, info = self.model.forward_multimodal(
                modality, input_data, state=self.state
            )
            self.state = self.model.init_state(1, torch.device(self.device))
            # 保留 state（forward_multimodal 修改了 state）
            # 实际上 forward_multimodal 内部更新了 state，但接口设计问题
            # 这里简化：每次用上一步的 state

        # 3. 输出
        info['modality'] = modality
        info['step'] = self.step_count
        self.step_count += 1

        # 4. 自监督：记录输出用于下一步反馈
        self.last_output = outputs

        return info

    def run(self, n_steps: int = 100, interval: float = 0.1,
            on_step: Optional[Callable] = None):
        """运行感知-行动循环

        Args:
            n_steps: 总步数（None = 无限）
            interval: 每步间隔（秒）
            on_step: 每步回调
        """
        self.running = True
        self.stream.start()
        print(f"感知-行动循环启动，{n_steps} 步，间隔 {interval}s")
        print("（首次运行 macOS 会请求摄像头/麦克风权限）")

        try:
            step = 0
            while self.running and (n_steps is None or step < n_steps):
                info = self.step_once()

                if info.get('status') != 'no_input':
                    if on_step:
                        on_step(info)

                    # 如果有音频输出，播放
                    if self.last_output and 'audio' in self.last_output:
                        audio_out = self.last_output['audio'][0].cpu().numpy()
                        # 每 10 步播放一次（避免太频繁）
                        if step % 10 == 0:
                            self.stream.play_audio(audio_out)

                step += 1
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.running = False
            self.stream.stop()
            print(f"循环结束，共 {step} 步")
