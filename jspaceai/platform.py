"""
平台抽象层——跨平台支持（macOS / Windows / Linux）

统一封装各平台差异：屏幕尺寸、权限检查、路径等。
所有平台相关代码集中在这里。
"""
from __future__ import annotations

import sys
import platform
from dataclasses import dataclass


@dataclass
class PlatformInfo:
    """平台信息"""
    system: str
    machine: str
    python_version: str
    is_macos: bool
    is_windows: bool
    is_linux: bool

    @classmethod
    def detect(cls) -> "PlatformInfo":
        s = platform.system()
        return cls(
            system=s,
            machine=platform.machine(),
            python_version=sys.version,
            is_macos=(s == 'Darwin'),
            is_windows=(s == 'Windows'),
            is_linux=(s == 'Linux'),
        )

    def __str__(self) -> str:
        return f"{self.system}/{self.machine} (Python {self.python_version.split()[0]})"


PLATFORM = PlatformInfo.detect()


def get_screen_size() -> tuple[int, int]:
    """获取主屏幕尺寸（跨平台）"""
    try:
        import mss
        with mss.MSS() as sct:
            mon = sct.monitors[1]
            return (mon['width'], mon['height'])
    except Exception:
        pass

    if PLATFORM.is_windows:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        except Exception:
            pass

    return (1920, 1080)


def check_camera_permission() -> bool:
    """检查摄像头权限"""
    if not PLATFORM.is_macos:
        return True
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


def check_microphone_permission() -> bool:
    """检查麦克风权限"""
    try:
        import sounddevice as sd
        sd.query_devices(kind='input')
        return True
    except Exception:
        return False


def check_input_monitoring_permission() -> bool:
    """检查键盘/鼠标监听权限"""
    if PLATFORM.is_windows:
        return True
    if PLATFORM.is_linux:
        return True
    if PLATFORM.is_macos:
        try:
            import subprocess
            result = subprocess.run(
                ['osascript', '-e',
                 'tell application "System Events" to keystroke ""'],
                capture_output=True, timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False
    return True


def print_permission_guide():
    """打印权限配置指南"""
    print("\n权限配置指南:")
    print("-" * 50)
    if PLATFORM.is_macos:
        print("macOS 权限设置:")
        print("  1. 摄像头: 系统设置 → 隐私与安全 → 摄像头")
        print("  2. 麦克风: 系统设置 → 隐私与安全 → 麦克风")
        print("  3. 键盘/鼠标监听: 系统设置 → 隐私与安全 → 辅助功能")
        print("  4. 屏幕录制: 系统设置 → 隐私与安全 → 屏幕录制")
    elif PLATFORM.is_linux:
        print("Linux 权限设置:")
        print("  1. 摄像头: 确保用户在 video 组 (sudo usermod -aG video $USER)")
        print("  2. 音频: 确保用户在 audio 组")
        print("  3. 键盘/鼠标: 需要 X11 或 Wayland 输入权限")
        print("  4. 屏幕捕获: 需要 X11 或安装 grim/slurp (Wayland)")
    elif PLATFORM.is_windows:
        print("Windows 权限设置:")
        print("  1. 摄像头/麦克风: 设置 → 隐私 → 摄像头/麦克风")
        print("  2. 键盘/鼠标: 通常不需要额外权限")
    print("-" * 50)
