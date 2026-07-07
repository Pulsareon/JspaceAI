#!/usr/bin/env python3
"""JspaceAI 自主心智 demo —— 永不停止的自主进化

运行：
    python main_autonomous.py --steps 100
    # 再次运行会从上次状态继续
    python main_autonomous.py --steps 100
"""
from __future__ import annotations
import argparse, torch, numpy as np, matplotlib.pyplot as plt
from jspaceai import (
    MultimodalConfig, MultimodalJSpaceModel, EmbodiedAgent,
    AutonomousMind, PLATFORM,
)


def get_config():
    return MultimodalConfig(
        vocab_size=50, embed_dim=16, input_dim=8, workspace_dim=64,
        expert_dim=24, num_experts=12, num_wells=4, ode_steps=3,
        dt=0.1, tau_w=0.3, jacobian_sparsity=16, noise_std=0.01,
        img_size=32, audio_frame_size=1024, keyboard_vocab=128,
    )


def main():
    p = argparse.ArgumentParser(description='JspaceAI 自主心智')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--device', default='cpu')
    p.add_argument('--unsafe', action='store_true')
    args = p.parse_args()

    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"平台: {PLATFORM}")
    config = get_config()
    model = MultimodalJSpaceModel(config).to(dev)
    model.eval()

    agent = EmbodiedAgent(
        model, device=dev,
        enable_mouse_output=args.unsafe,
        enable_keyboard_output=args.unsafe,
        enable_audio_output=True, enable_screen_output=False,
    )
    mind = AutonomousMind(agent, save_dir='outputs/mind', device=dev)

    print("\n" + "=" * 60)
    print("自主心智 - 永不停止的进化")
    print("=" * 60)
    print("1. 好奇心驱动  2. 状态持久化  3. 自我模型  4. 元学习")
    print(f"运行 {args.steps} 步（Ctrl+C 中断，状态自动保存）\n")

    log = []

    def on_step(info):
        log.append(info)
        if info['step'] % 10 == 0:
            print(f"  step {info['step']:4d} | mod {info['modality']:8s} | "
                  f"||w|| {info['w_norm']:.3f} | curio {info['curiosity']:.3f} | "
                  f"success {info['success']:.2f} | weak={info['weakness']}")

    mind.run(n_steps=args.steps, interval=0.2, save_every=30, on_step=on_step)
    print("\n" + mind.introspect())

    if log:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        steps = [s['step'] for s in log]

        axes[0, 0].plot(steps, [s['w_norm'] for s in log], 'b-')
        axes[0, 0].set_title('Workspace ||w||'); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, [s['curiosity'] for s in log], 'r-', label='curiosity')
        axes[0, 1].plot(steps, [s['success'] for s in log], 'g-', label='success')
        axes[0, 1].legend(); axes[0, 1].set_title('Curiosity & Success')
        axes[0, 1].grid(True, alpha=0.3)

        fc = log[-1]['self_confidence']
        axes[1, 0].barh(list(fc.keys()), list(fc.values()),
                        color=plt.cm.RdYlGn(list(fc.values())))
        axes[1, 0].set_xlim(0, 1); axes[1, 0].set_title('Self Model')

        axes[1, 1].plot(steps, [s['world_loss'] for s in log], 'orange')
        axes[1, 1].set_title('World Model Loss'); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig('outputs/autonomous_mind.png', dpi=150, bbox_inches='tight')
        print(f"\n可视化: outputs/autonomous_mind.png")


if __name__ == '__main__':
    from pathlib import Path
    main()
