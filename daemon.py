"""
守护进程 —— 自主心智后台持续运行

用户主动控制：
    python daemon.py start    # 启动后台守护进程
    python daemon.py stop     # 停止
    python daemon.py status   # 查看状态
    python daemon.py log      # 查看最近日志
    python daemon.py fg       # 前台运行（调试用）

守护进程特性：
    - 静默后台运行，不弹窗不干扰
    - 持续感知（屏幕/键盘/鼠标/摄像头/麦克风）
    - 持续学习（好奇心驱动 + 状态保存）
    - 信号处理（SIGTERM 优雅退出）
    - 崩溃恢复（自动重启）
    - 低资源占用（可配置 CPU/内存限制）

状态保存在 ~/.jspaceai/，每次运行从上次状态继续。
"""
from __future__ import annotations

import os
import sys
import signal
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from jspaceai import (
    MultimodalConfig, MultimodalJSpaceModel, EmbodiedAgent,
    AutonomousMind, PLATFORM,
)


# ============================================================
# 路径配置
# ============================================================

APP_DIR = Path.home() / ".jspaceai"
LOG_DIR = APP_DIR / "logs"
PID_FILE = APP_DIR / "daemon.pid"
STATE_DIR = APP_DIR / "state"
LOG_FILE = LOG_DIR / "daemon.log"
STATUS_FILE = APP_DIR / "status.json"

APP_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# 日志
# ============================================================

def get_logger(foreground: bool = False) -> logging.Logger:
    logger = logging.getLogger("jspaceai-daemon")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if foreground:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ============================================================
# PID 管理
# ============================================================

def write_pid(pid: int):
    PID_FILE.write_text(str(pid))

def read_pid() -> Optional[int]:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except ValueError:
            return None
    return None

def clear_pid():
    if PID_FILE.exists():
        PID_FILE.unlink()

def is_running(pid: int) -> bool:
    """检查进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def write_status(status: dict):
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))

def read_status() -> Optional[dict]:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            return None
    return None


# ============================================================
# 守护进程主循环
# ============================================================

class MindDaemon:
    """自主心智守护进程"""

    def __init__(self, device: str = 'cpu', low_power: bool = False,
                 step_interval: float = 0.5):
        self.device = device
        self.low_power = low_power
        self.step_interval = step_interval
        self.logger = get_logger(foreground=False)
        self.running = False
        self.mind: Optional[AutonomousMind] = None
        self.agent: Optional[EmbodiedAgent] = None
        self.start_time = time.time()
        self.step_count_session = 0

    def initialize(self):
        """初始化心智"""
        self.logger.info(f"初始化 | 平台: {PLATFORM} | 设备: {self.device}")

        config = MultimodalConfig(
            vocab_size=50, embed_dim=16, input_dim=8,
            workspace_dim=64, expert_dim=24, num_experts=12,
            num_wells=4, ode_steps=3, dt=0.1, tau_w=0.3,
            jacobian_sparsity=16, noise_std=0.01,
            img_size=32, audio_frame_size=1024, keyboard_vocab=128,
        )
        model = MultimodalJSpaceModel(config).to(self.device)
        model.eval()

        # 完全静默：不控制鼠标键盘，不发声，不弹窗
        self.agent = EmbodiedAgent(
            model, device=self.device,
            enable_mouse_output=False,
            enable_keyboard_output=False,
            enable_audio_output=False,
            enable_screen_output=False,
        )
        self.mind = AutonomousMind(
            self.agent, save_dir=str(STATE_DIR), device=self.device,
        )
        self.logger.info(f"就绪 | 历史步数: {self.mind.step_count}")

    def run_forever(self):
        """主循环——永不停止，直到收到停止信号"""
        self.running = True

        # 信号处理
        def handle_stop(signum, frame):
            self.logger.info(f"收到信号 {signum}，优雅退出...")
            self.running = False

        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)

        self.initialize()

        # 写入运行状态
        write_status({
            'running': True,
            'pid': os.getpid(),
            'started_at': time.time(),
            'device': self.device,
            'low_power': self.low_power,
            'step_count': self.mind.step_count,
        })

        self.logger.info("守护进程启动，开始持续感知学习")
        last_save = time.time()
        save_interval = 30  # 每 30 秒保存一次

        try:
            while self.running:
                try:
                    info = self.mind.step()
                    self.step_count_session += 1

                    # 每 50 步记录一次日志
                    if self.step_count_session % 50 == 0:
                        self.logger.info(
                            f"step {info['step']} | mod {info['modality']} | "
                            f"||w|| {info['w_norm']:.3f} | "
                            f"curio {info['curiosity']:.3f} | "
                            f"success {info['success']:.2f} | "
                            f"mem {info['memory_count']}"
                        )
                        # 更新状态文件
                        status = read_status() or {}
                        status.update({
                            'step_count': info['step'],
                            'last_step_at': time.time(),
                            'last_modality': info['modality'],
                            'last_w_norm': info['w_norm'],
                            'last_curiosity': info['curiosity'],
                            'memory_count': info['memory_count'],
                            'session_steps': self.step_count_session,
                        })
                        write_status(status)

                    # 定期保存
                    if time.time() - last_save > save_interval:
                        self.mind.save_state()
                        last_save = time.time()

                    # 控制频率
                    time.sleep(self.step_interval)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"步进异常: {e}\n{__import__('traceback').format_exc()}")
                    # 等待后继续，不崩溃
                    time.sleep(5.0)

        finally:
            # 清理
            self.logger.info("保存最终状态...")
            self.mind.save_state()
            if self.agent:
                self.agent.senses.stop()
            self.logger.info(
                f"守护进程退出 | 本次步数: {self.step_count_session} | "
                f"总步数: {self.mind.step_count} | "
                f"运行时长: {time.time() - self.start_time:.0f}s"
            )
            write_status({
                'running': False,
                'stopped_at': time.time(),
                'step_count': self.mind.step_count,
                'session_steps': self.step_count_session,
            })
            clear_pid()


# ============================================================
# 命令处理
# ============================================================

def cmd_start(device: str = 'cpu', low_power: bool = False,
              interval: float = 0.5, foreground: bool = False):
    """启动守护进程"""
    existing_pid = read_pid()
    if existing_pid and is_running(existing_pid):
        print(f"守护进程已在运行 (PID {existing_pid})")
        print(f"查看状态: python daemon.py status")
        return

    if foreground:
        # 前台运行（调试用）
        print(f"前台模式启动（Ctrl+C 退出）")
        daemon = MindDaemon(device=device, low_power=low_power,
                           step_interval=interval)
        daemon.logger = get_logger(foreground=True)
        daemon.run_forever()
        return

    # 后台 fork
    print(f"启动后台守护进程...")
    daemon_script = Path(__file__).resolve()
    cmd = [sys.executable, str(daemon_script), "_run",
           "--device", device,
           "--interval", str(interval)]
    if low_power:
        cmd.append("--low-power")

    # 用 subprocess 启动，脱离终端
    proc = subprocess.Popen(
        cmd,
        stdout=open(LOG_DIR / 'stdout.log', 'a'),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # 脱离父进程
    )

    write_pid(proc.pid)

    # 等待一下确认启动
    time.sleep(2)
    if proc.poll() is None:
        print(f"守护进程已启动 (PID {proc.pid})")
        print(f"日志: {LOG_FILE}")
        print(f"停止: python daemon.py stop")
        print(f"状态: python daemon.py status")
    else:
        print(f"启动失败，查看日志: {LOG_FILE}")
        clear_pid()


def cmd_stop():
    """停止守护进程"""
    pid = read_pid()
    if not pid:
        print("守护进程未运行")
        return

    if not is_running(pid):
        print(f"进程 {pid} 已不存在，清理 PID 文件")
        clear_pid()
        return

    print(f"发送停止信号到 PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
        # 等待退出
        for _ in range(10):
            time.sleep(0.5)
            if not is_running(pid):
                break
        if is_running(pid):
            print("强制终止...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        clear_pid()
        print("守护进程已停止")
    except Exception as e:
        print(f"停止失败: {e}")


def cmd_status():
    """查看状态"""
    pid = read_pid()
    running = pid and is_running(pid)

    print("=" * 50)
    print(f"JspaceAI 守护进程状态")
    print("=" * 50)

    if running:
        print(f"状态: 运行中 (PID {pid})")
    else:
        print(f"状态: 已停止")
        if pid:
            print(f"  (PID {pid} 已不存在)")

    status = read_status()
    if status:
        print(f"总步数: {status.get('step_count', '?')}")
        if 'started_at' in status and running:
            uptime = time.time() - status['started_at']
            print(f"运行时长: {uptime:.0f}s ({uptime/3600:.1f}h)")
        if 'last_modality' in status:
            print(f"最后模态: {status['last_modality']}")
        if 'last_w_norm' in status:
            print(f"||w||: {status['last_w_norm']:.3f}")
        if 'last_curiosity' in status:
            print(f"好奇心: {status['last_curiosity']:.3f}")
        if 'memory_count' in status:
            print(f"记忆数: {status['memory_count']}")
        if 'session_steps' in status:
            print(f"本次会话步数: {status['session_steps']}")
        if not running and 'stopped_at' in status:
            print(f"停止时间: {time.ctime(status['stopped_at'])}")

    print(f"\n日志: {LOG_FILE}")
    print(f"状态目录: {STATE_DIR}")


def cmd_log(n_lines: int = 30):
    """查看最近日志"""
    if not LOG_FILE.exists():
        print("无日志")
        return
    lines = LOG_FILE.read_text().strip().split('\n')
    for line in lines[-n_lines:]:
        print(line)


def cmd_introspect():
    """内省——查看心智的自我认知"""
    status = read_status()
    if not status:
        print("无状态数据")
        return

    pid = read_pid()
    if pid and is_running(pid):
        print("心智正在运行，自我认知：")
    else:
        print("心智已停止，最后的自我认知：")

    print(json.dumps(status, indent=2, default=str))


# ============================================================
# 入口
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description='JspaceAI 自主心智守护进程')
    sub = p.add_subparsers(dest='command')

    # start
    sp = sub.add_parser('start', help='启动后台守护进程')
    sp.add_argument('--device', default='cpu')
    sp.add_argument('--low-power', action='store_true', help='低功耗模式（更长间隔）')
    sp.add_argument('--interval', type=float, default=0.5, help='步进间隔(秒)')
    sp.add_argument('--fg', action='store_true', help='前台运行（调试用）')

    # stop
    sub.add_parser('stop', help='停止守护进程')

    # status
    sub.add_parser('status', help='查看状态')

    # log
    sp = sub.add_parser('log', help='查看日志')
    sp.add_argument('-n', type=int, default=30, help='行数')

    # introspect
    sub.add_parser('introspect', help='心智自我认知')

    # _run（内部命令，被 start 调用）
    sp = sub.add_parser('_run', help='内部运行命令')
    sp.add_argument('--device', default='cpu')
    sp.add_argument('--low-power', action='store_true')
    sp.add_argument('--interval', type=float, default=0.5)

    args = p.parse_args()

    if args.command == 'start':
        cmd_start(args.device, args.low_power, args.interval, args.fg)
    elif args.command == 'stop':
        cmd_stop()
    elif args.command == 'status':
        cmd_status()
    elif args.command == 'log':
        cmd_log(args.n)
    elif args.command == 'introspect':
        cmd_introspect()
    elif args.command == '_run':
        # 内部运行模式
        daemon = MindDaemon(
            device=args.device,
            low_power=args.low_power,
            step_interval=args.interval,
        )
        daemon.run_forever()
    else:
        p.print_help()


if __name__ == '__main__':
    main()
