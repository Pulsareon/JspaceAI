"""
屏幕 + 键盘 + 鼠标 I/O 层

使用 mss（屏幕捕获）+ pynput（键盘鼠标监听）。

提供 ScreenCapture / KeyboardMonitor / MouseMonitor / DesktopStream / FullSensoryStream
"""
from __future__ import annotations

import mss
import numpy as np
import torch
import threading
import queue
import time
from typing import Optional
from dataclasses import dataclass, field
from pynput import keyboard, mouse
from PIL import Image as PILImage


@dataclass
class InputEvent:
    """统一的输入事件"""
    timestamp: float
    device: str
    event_type: str
    data: object
    modifiers: dict = field(default_factory=dict)


class ScreenCapture:
    """屏幕捕获流"""

    def __init__(self, target_size: tuple = (32, 32), fps: int = 5):
        self.target_size = target_size
        self.fps = fps
        self.frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._sct = None

    def start(self):
        self._sct = mss.MSS()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        interval = 1.0 / self.fps
        while self.running:
            try:
                monitor = self._sct.monitors[1]
                raw = self._sct.grab(monitor)
                img = np.array(raw)[:, :, :3]
                pil_img = PILImage.fromarray(img)
                pil_img = pil_img.resize(self.target_size, PILImage.BILINEAR)
                img_resized = np.array(pil_img)
                self.frame_queue.put_nowait(
                    InputEvent(time.time(), 'screen', 'frame', img_resized)
                )
            except queue.Full:
                pass
            except Exception:
                pass
            time.sleep(interval)

    def get_frame(self) -> Optional[InputEvent]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self._sct:
            self._sct.close()


class KeyboardMonitor:
    """键盘监听"""

    def __init__(self):
        self.event_queue: queue.Queue = queue.Queue(maxsize=200)
        self.running = False
        self.listener: Optional[keyboard.Listener] = None
        self.buffer: list[str] = []

    def start(self):
        self.running = True
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        if not self.running:
            return
        try:
            char = key.char
            self.buffer.append(char)
            self.event_queue.put_nowait(
                InputEvent(time.time(), 'keyboard', 'key', char, {'action': 'press'})
            )
        except AttributeError:
            name = str(key).replace('Key.', '')
            self.event_queue.put_nowait(
                InputEvent(time.time(), 'keyboard', 'key', name,
                          {'action': 'press', 'special': True})
            )

    def get_events(self) -> list[InputEvent]:
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def get_buffer_text(self) -> str:
        return ''.join(self.buffer)

    def clear_buffer(self):
        self.buffer.clear()

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()


class MouseMonitor:
    """鼠标监听"""

    def __init__(self):
        self.event_queue: queue.Queue = queue.Queue(maxsize=200)
        self.running = False
        self.listener: Optional[mouse.Listener] = None
        self.last_pos: tuple[int, int] = (0, 0)

    def start(self):
        self.running = True
        self.listener = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll,
        )
        self.listener.start()

    def _on_move(self, x, y):
        if not self.running:
            return
        self.last_pos = (x, y)
        try:
            self.event_queue.put_nowait(
                InputEvent(time.time(), 'mouse', 'move', (x, y))
            )
        except queue.Full:
            pass

    def _on_click(self, x, y, button, pressed):
        if not self.running:
            return
        btn = 'left' if button == mouse.Button.left else 'right'
        action = 'click' if pressed else 'release'
        try:
            self.event_queue.put_nowait(
                InputEvent(time.time(), 'mouse', 'click', (x, y),
                          {'button': btn, 'action': action})
            )
        except queue.Full:
            pass

    def _on_scroll(self, x, y, dx, dy):
        if not self.running:
            return
        try:
            self.event_queue.put_nowait(
                InputEvent(time.time(), 'mouse', 'scroll', (x, y),
                          {'dx': dx, 'dy': dy})
            )
        except queue.Full:
            pass

    def get_events(self) -> list[InputEvent]:
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def get_position(self) -> tuple[int, int]:
        return self.last_pos

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()


class DesktopStream:
    """统一管理屏幕 + 键盘 + 鼠标"""

    def __init__(self, use_screen: bool = True, use_keyboard: bool = True,
                 use_mouse: bool = True, screen_size: tuple = (32, 32),
                 screen_fps: int = 5):
        self.use_screen = use_screen
        self.use_keyboard = use_keyboard
        self.use_mouse = use_mouse
        self.screen = ScreenCapture(target_size=screen_size, fps=screen_fps) if use_screen else None
        self.keyboard = KeyboardMonitor() if use_keyboard else None
        self.mouse = MouseMonitor() if use_mouse else None

    def start(self):
        if self.screen:
            try:
                self.screen.start()
                print("  屏幕捕获已启动")
            except Exception as e:
                print(f"  屏幕捕获失败: {e}")
                self.screen = None

        if self.keyboard:
            try:
                self.keyboard.start()
                print("  键盘监听已启动")
            except Exception as e:
                print(f"  键盘监听失败: {e}")
                self.keyboard = None

        if self.mouse:
            try:
                self.mouse.start()
                print("  鼠标监听已启动")
            except Exception as e:
                print(f"  鼠标监听失败: {e}")
                self.mouse = None

    def get_latest(self) -> dict:
        result = {
            'screen': None, 'keyboard': [], 'mouse': [],
            'mouse_pos': (0, 0), 'keyboard_buffer': '',
        }
        if self.screen:
            latest = None
            while True:
                f = self.screen.get_frame()
                if f is None:
                    break
                latest = f
            result['screen'] = latest

        if self.keyboard:
            result['keyboard'] = self.keyboard.get_events()
            result['keyboard_buffer'] = self.keyboard.get_buffer_text()

        if self.mouse:
            result['mouse'] = self.mouse.get_events()
            result['mouse_pos'] = self.mouse.get_position()

        return result

    def stop(self):
        if self.screen:
            self.screen.stop()
        if self.keyboard:
            self.keyboard.stop()
        if self.mouse:
            self.mouse.stop()


class FullSensoryStream:
    """
    完整感知流：摄像头 + 麦克风 + 屏幕 + 键盘 + 鼠标。

    五个通道并行采集，这是模型感知世界的全部输入。

    通道优先级（当多个同时有数据时）：
        1. 键盘（用户主动输入，最高）
        2. 鼠标点击
        3. 音频（麦克风）
        4. 屏幕（用户在看什么）
        5. 摄像头（环境）
    """

    def __init__(self, use_camera: bool = True, use_mic: bool = True,
                 use_desktop: bool = True, img_size: tuple = (32, 32),
                 sample_rate: int = 16000, audio_frame_size: int = 1024):
        from .realtime import MultimodalStream

        self.av_stream = MultimodalStream(
            use_camera=use_camera, use_mic=use_mic,
            img_size=img_size, sample_rate=sample_rate,
            audio_frame_size=audio_frame_size,
        ) if use_camera or use_mic else None

        self.desktop = DesktopStream(
            use_screen=use_desktop, use_keyboard=use_desktop, use_mouse=use_desktop,
            screen_size=img_size,
        ) if use_desktop else None

    def start(self):
        if self.av_stream:
            self.av_stream.start()
        if self.desktop:
            self.desktop.start()

    def get_latest(self) -> dict:
        """获取所有通道的最新数据"""
        result = {
            'camera': None, 'audio': None,
            'screen': None, 'keyboard': [], 'mouse': [],
            'mouse_pos': (0, 0), 'keyboard_buffer': '',
        }
        if self.av_stream:
            av_frames = self.av_stream.get_latest_frames()
            result['camera'] = av_frames.get('image')
            result['audio'] = av_frames.get('audio')

        if self.desktop:
            desk = self.desktop.get_latest()
            result.update(desk)

        return result

    def play_audio(self, audio: np.ndarray, blocking: bool = False):
        if self.av_stream:
            self.av_stream.play_audio(audio, blocking)

    def stop(self):
        if self.av_stream:
            self.av_stream.stop()
        if self.desktop:
            self.desktop.stop()
