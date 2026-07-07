#!/usr/bin/env python3
"""
JspaceAI —— 全部接入版（具身 Agent + 完整神经系统 + 自主心智）

接入全部 5 个感官输入 + 4 个输出执行器 + 神经系统（小脑/海马体/基底神经节/中枢神经）
+ 自主心智（好奇心驱动 + 状态持久化 + 自我模型 + 元学习）。

模式：
    --mode test:    子系统自检（权限 + I/O + 模型 + 海马体回忆）
    --mode live:    实时具身循环 + 自主心智（默认安全：不控制鼠标键盘）
    --mode safe:    同 live 但更保守（更高动作门控阈值）

运行：
    python main.py --mode test
    python main.py --mode live --steps 100
    python main.py --mode safe --steps 50
    python main.py --mode live --steps 200 --unsafe   # 允许鼠标键盘输出

再次运行会从上次状态继续（海洋不蒸发）。
"""
from __future__ import annotations
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from jspaceai import (
    MultimodalConfig, MultimodalJSpaceModel, EmbodiedAgent,
    AutonomousMind, PLATFORM,
    get_screen_size, print_permission_guide,
    check_camera_permission, check_microphone_permission,
    check_input_monitoring_permission,
)


def get_config() -> MultimodalConfig:
    return MultimodalConfig(
        vocab_size=50, embed_dim=16, input_dim=8, workspace_dim=64,
        expert_dim=24, num_experts=12, num_wells=4, ode_steps=3,
        dt=0.1, tau_w=0.3, jacobian_sparsity=16, noise_std=0.01,
        img_size=32, audio_frame_size=1024, keyboard_vocab=128,
    )


def test_subsystems():
    """子系统自检"""
    print("=" * 60)
    print(f"具身 Agent 子系统测试 | 平台: {PLATFORM}")
    print("=" * 60)

    print("\n1. 权限检查:")
    checks = {
        '摄像头': check_camera_permission(),
        '麦克风': check_microphone_permission(),
        '键盘/鼠标监听': check_input_monitoring_permission(),
    }
    for name, ok in checks.items():
        print(f"   {name}: {'OK' if ok else '需要权限'}")
    if not all(checks.values()):
        print_permission_guide()

    sw, sh = get_screen_size()
    print(f"   屏幕尺寸: {sw}x{sh}")

    print("\n2. 模型 + Agent 初始化:")
    config = get_config()
    model = MultimodalJSpaceModel(config)
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   专家分工: {model.expert_modality}")

    agent = EmbodiedAgent(
        model, device='cpu',
        enable_mouse_output=False, enable_keyboard_output=False,
        enable_audio_output=True, enable_screen_output=False,
    )
    print(f"   小脑参数: {sum(p.numel() for p in agent.cerebellum.parameters()):,}")
    print(f"   海马体容量: {agent.hippocampus.capacity}")
    print(f"   反射弧数: {len(agent.cns.reflexes)}")

    print("\n3. 单步循环测试（2秒采集）:")
    agent.senses.start()
    time.sleep(2)
    for _ in range(5):
        info = agent.step_once()
        print(f"   step {info['step']:2d} | mod {info['modality']:8s} | "
              f"||w|| {info['w_norm']:.3f} | action {info['action']['action_idx']} | "
              f"executed {info['action']['executed']} | mem {info['memories_count']}")
        time.sleep(0.5)
    agent.senses.stop()
    agent.audio_actuator.stop()

    print("\n4. 海马体回忆测试:")
    if agent.hippocampus and agent.hippocampus.size() > 0:
        w = agent.state['w'][0].cpu().numpy()
        for i, m in enumerate(agent.hippocampus.recall(w, top_k=3)):
            print(f"   记忆 {i}: sim={m['similarity']:.3f} ctx={m['context']}")

    print("\n所有子系统测试完成")


def live(n_steps: int, device: str, safe_mode: bool = False, unsafe: bool = False):
    """实时具身循环 + 自主心智"""
    print(f"平台: {PLATFORM}")
    sw, sh = get_screen_size()
    print(f"屏幕: {sw}x{sh}")

    allow_output = unsafe and not safe_mode
    if safe_mode:
        print("\n安全模式：不执行鼠标/键盘动作")
        print_permission_guide()
    elif not allow_output:
        print("\n默认模式：仅感知 + 音频/屏幕输出（不控制鼠标键盘）")
        print("  如需鼠标键盘输出，加 --unsafe")

    config = get_config()
    model = MultimodalJSpaceModel(config).to(device)
    mp = Path('outputs/multimodal_model.pt')
    if mp.exists():
        try:
            ckpt = torch.load(mp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=False)
            print(f"已加载模型: {mp}")
        except Exception:
            print("模型加载失败，随机初始化")
    else:
        print("未找到训练模型，随机初始化")
    model.eval()

    agent = EmbodiedAgent(
        model, device=device,
        enable_mouse_output=allow_output,
        enable_keyboard_output=allow_output,
        enable_audio_output=True,
        enable_screen_output=not safe_mode,
        risk_threshold=0.5 if safe_mode else 0.3,
    )
    mind = AutonomousMind(agent, save_dir='outputs/mind', device=device)

    print("\n" + "=" * 60)
    print("自主心智 - 全感官具身循环")
    print("=" * 60)
    print("好奇心驱动 + 状态持久化 + 自我模型 + 元学习")
    print(f"运行 {n_steps} 步（Ctrl+C 中断，状态自动保存）\n")

    log = []

    def on_step(info):
        log.append(info)
        if info['step'] % 10 == 0:
            print(f"  step {info['step']:4d} | mod {info['modality']:8s} | "
                  f"||w|| {info['w_norm']:.3f} | curio {info['curiosity']:.3f} | "
                  f"success {info['success']:.2f} | weak={info['weakness']} | "
                  f"mem {info['memory_count']}")

    mind.run(n_steps=n_steps, interval=0.2, save_every=30, on_step=on_step)

    print("\n" + mind.introspect())

    # 可视化
    if log:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        steps = [s['step'] for s in log]

        # ||w|| 按模态着色
        ax = axes[0, 0]
        wn = [s['w_norm'] for s in log]
        mods = [s['modality'] for s in log]
        cm = {'image': 'green', 'screen': 'purple', 'audio': 'blue',
              'keyboard': 'orange', 'mouse': 'red', 'idle': 'gray'}
        colors = [cm.get(m, 'gray') for m in mods]
        ax.scatter(steps, wn, c=colors, alpha=0.7, s=30)
        ax.set_xlabel('Step'); ax.set_ylabel('||w||')
        ax.set_title('Workspace Norm (by modality)'); ax.grid(True, alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in cm.items()],
                  fontsize=7)

        # 好奇心 + 成功度
        ax = axes[0, 1]
        ax.plot(steps, [s['curiosity'] for s in log], 'r-', label='curiosity')
        ax.plot(steps, [s['success'] for s in log], 'g-', label='success')
        ax.legend(); ax.set_title('Curiosity & Success')
        ax.grid(True, alpha=0.3)

        # 模态分布
        ax = axes[0, 2]
        mc = {}
        for m in mods:
            mc[m] = mc.get(m, 0) + 1
        ax.bar(mc.keys(), mc.values(),
               color=[cm.get(m, 'gray') for m in mc.keys()])
        ax.set_title('Modality Distribution')

        # 自我模型
        ax = axes[1, 0]
        fc = log[-1]['self_confidence']
        ax.barh(list(fc.keys()), list(fc.values()),
                color=plt.cm.RdYlGn(list(fc.values())))
        ax.set_xlim(0, 1); ax.set_title('Self Model (confidence)')

        # 世界模型 loss
        ax = axes[1, 1]
        ax.plot(steps, [s['world_loss'] for s in log], 'orange')
        ax.set_title('World Model Loss'); ax.grid(True, alpha=0.3)

        # 记忆数
        ax = axes[1, 2]
        ax.plot(steps, [s['memory_count'] for s in log], 'teal')
        ax.set_title('Hippocampus Memory Count'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig('outputs/embodied_live.png', dpi=150, bbox_inches='tight')
        print(f"\n可视化: outputs/embodied_live.png")


def main():
    p = argparse.ArgumentParser(description='JspaceAI 全部接入版（具身 + 自主心智）')
    p.add_argument('--mode', default='test', choices=['test', 'live', 'safe'],
                   help='运行模式: test=自检, live=实时循环, safe=安全模式')
    p.add_argument('--steps', type=int, default=100, help='live/safe 模式步数')
    p.add_argument('--device', default='cpu', help='设备 (cpu/cuda/mps/auto)')
    p.add_argument('--unsafe', action='store_true',
                   help='允许鼠标键盘输出（默认禁用）')
    args = p.parse_args()

    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')

    if args.mode == 'test':
        test_subsystems()
    elif args.mode == 'live':
        live(args.steps, dev, safe_mode=False, unsafe=args.unsafe)
    elif args.mode == 'safe':
        live(args.steps, dev, safe_mode=True, unsafe=False)


if __name__ == '__main__':
    main()
