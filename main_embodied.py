#!/usr/bin/env python3
"""JspaceAI 具身 Agent - 完整神经系统 + 跨平台"""
from __future__ import annotations
import argparse, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import time
from jspaceai import (
    MultimodalConfig, MultimodalJSpaceModel, EmbodiedAgent,
    PLATFORM, get_screen_size, print_permission_guide,
    check_camera_permission, check_microphone_permission,
    check_input_monitoring_permission,
)


def get_config():
    return MultimodalConfig(
        vocab_size=50, embed_dim=16, input_dim=8, workspace_dim=64,
        expert_dim=24, num_experts=12, num_wells=4, ode_steps=3,
        dt=0.1, tau_w=0.3, jacobian_sparsity=16, noise_std=0.01,
        img_size=32, audio_frame_size=1024, keyboard_vocab=128,
    )


def test_subsystems():
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
    for i in range(5):
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


def live_embodied(n_steps, device, safe_mode=False):
    print(f"平台: {PLATFORM}")
    sw, sh = get_screen_size()
    print(f"屏幕: {sw}x{sh}")
    if safe_mode:
        print("\n安全模式：不执行鼠标/键盘动作")
        print_permission_guide()

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
        enable_mouse_output=not safe_mode,
        enable_keyboard_output=not safe_mode,
        enable_audio_output=True,
        enable_screen_output=not safe_mode,
        risk_threshold=0.5 if safe_mode else 0.3,
    )

    log = []

    def on_step(info):
        log.append(info)
        if info['step'] % 5 == 0:
            a = info['action']
            print(f"  step {info['step']:3d} | mod {info['modality']:8s} | "
                  f"||w|| {info['w_norm']:.3f} | action {a['action_idx']} "
                  f"str {a['action_strength']:.2f} "
                  f"exec {'Y' if a['executed'] else 'N'} | "
                  f"mem {info['memories_count']}")

    print(f"\n运行 {n_steps} 步具身循环...")
    agent.run(n_steps=n_steps, interval=0.2, on_step=on_step)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"总步数: {len(log)}")
    if log:
        wn = [s['w_norm'] for s in log]
        print(f"||w||: [{min(wn):.3f}, {max(wn):.3f}] mean={np.mean(wn):.3f}")
        mc = {}
        for s in log:
            mc[s['modality']] = mc.get(s['modality'], 0) + 1
        print("\n模态分布:")
        for m, c in sorted(mc.items(), key=lambda x: -x[1]):
            print(f"  {m:10s}: {c:3d} ({c/len(log)*100:.0f}%)")
        ex = sum(1 for s in log if s['action']['executed'])
        print(f"\n动作执行: {ex}/{len(log)} ({ex/len(log)*100:.0f}%)")
        print("\n基底神经节:")
        for i, c in enumerate(agent.basal_ganglia.habit_counts):
            h = " (习惯化)" if agent.basal_ganglia.is_habitual(i) else ""
            print(f"  动作{i}: {int(c):4d}{h}")
        if agent.hippocampus:
            print(f"\n海马体: {agent.hippocampus.size()} 条记忆")

    if log:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        steps = [s['step'] for s in log]
        wn = [s['w_norm'] for s in log]
        mods = [s['modality'] for s in log]
        cm = {'image': 'green', 'screen': 'purple', 'audio': 'blue',
              'keyboard': 'orange', 'mouse': 'red', 'idle': 'gray'}
        colors = [cm.get(m, 'gray') for m in mods]

        ax = axes[0, 0]
        ax.scatter(steps, wn, c=colors, alpha=0.7, s=30)
        ax.set_xlabel('Step'); ax.set_ylabel('||w||')
        ax.set_title('Workspace Norm'); ax.grid(True, alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in cm.items()],
                  fontsize=7)

        ax = axes[0, 1]
        mc2 = {}
        for m in mods: mc2[m] = mc2.get(m, 0) + 1
        ax.bar(mc2.keys(), mc2.values(),
               color=[cm.get(m, 'gray') for m in mc2.keys()])
        ax.set_title('Modality Distribution')

        ax = axes[1, 0]
        ast = [s['action']['action_strength'] for s in log]
        exf = [1 if s['action']['executed'] else 0 for s in log]
        ax.plot(steps, ast, label='strength', alpha=0.7)
        ax.scatter(steps, exf, c=['green' if e else 'red' for e in exf],
                   label='executed', alpha=0.5, s=20)
        ax.set_title('Action Strength & Execution'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ai = [s['action']['action_idx'] for s in log]
        ax.hist(ai, bins=range(6), align='left', rwidth=0.8, color='steelblue', alpha=0.7)
        ax.set_title('Basal Ganglia Action Selection')
        ax.set_xticks(range(5))

        plt.tight_layout()
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig('outputs/embodied_live.png', dpi=150, bbox_inches='tight')
        print(f"\n可视化: outputs/embodied_live.png")


def main():
    p = argparse.ArgumentParser(description='JspaceAI 具身 Agent')
    p.add_argument('--mode', default='test', choices=['test', 'live', 'safe'])
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')

    if args.mode == 'test':
        test_subsystems()
    elif args.mode == 'live':
        live_embodied(args.steps, dev, safe_mode=False)
    elif args.mode == 'safe':
        live_embodied(args.steps, dev, safe_mode=True)


if __name__ == '__main__':
    main()
