#!/usr/bin/env python3
"""
JspaceAI 全感官交互 demo

接入全部 5 个输入通道：摄像头 + 麦克风 + 屏幕 + 键盘 + 鼠标

模式：
    --mode test:    测试各 I/O 通道是否工作
    --mode live:    实时全感官感知循环

运行：
    python main_full_sensory.py --mode test
    python main_full_sensory.py --mode live --steps 50
"""
from __future__ import annotations

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from jspaceai import (
    MultimodalConfig, MultimodalJSpaceModel,
    FullSensoryStream,
)
from jspaceai.platform import get_screen_size, PLATFORM, print_permission_guide


def get_config() -> MultimodalConfig:
    return MultimodalConfig(
        vocab_size=50, embed_dim=16, input_dim=8,
        workspace_dim=64, expert_dim=24, num_experts=12,
        num_wells=4, ode_steps=3, dt=0.1, tau_w=0.3,
        jacobian_sparsity=16, noise_std=0.01,
        img_size=32, audio_frame_size=1024, keyboard_vocab=128,
    )


def test_io():
    """测试所有 I/O 通道"""
    print("=" * 60)
    print("测试全感官 I/O 通道")
    print("=" * 60)

    stream = FullSensoryStream(
        use_camera=True, use_mic=True, use_desktop=True, img_size=(32, 32),
    )
    stream.start()

    print("\n采集 5 秒数据...（请移动鼠标、按键、对摄像头说话）")
    log = {'camera': 0, 'audio': 0, 'screen': 0, 'keyboard': 0, 'mouse': 0}
    mouse_positions = []

    for i in range(25):
        data = stream.get_latest()
        if data['camera']:
            log['camera'] += 1
        if data['audio']:
            log['audio'] += 1
        if data['screen']:
            log['screen'] += 1
        if data['keyboard']:
            log['keyboard'] += len(data['keyboard'])
        if data['mouse']:
            log['mouse'] += len(data['mouse'])
            mouse_positions.append(data['mouse_pos'])
        time.sleep(0.2)

    stream.stop()

    print("\n" + "=" * 60)
    print("采集结果")
    print("=" * 60)
    for ch, count in log.items():
        status = "OK" if count > 0 else "无数据"
        print(f"  {ch:10s}: {count:4d} 帧/事件  [{status}]")

    if mouse_positions:
        xs = [p[0] for p in mouse_positions]
        ys = [p[1] for p in mouse_positions]
        print(f"  鼠标位置范围: x=[{min(xs)}, {max(xs)}], y=[{min(ys)}, {max(ys)}]")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if mouse_positions:
        xs = [p[0] for p in mouse_positions]
        ys = [p[1] for p in mouse_positions]
        ax.plot(xs, ys, 'b.-', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Mouse Trajectory ({len(mouse_positions)} points)')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No mouse movement", transform=ax.transAxes, ha='center')
        ax.set_title('Mouse')

    ax = axes[1]
    channels = list(log.keys())
    counts = list(log.values())
    ax.bar(channels, counts, color=['green', 'blue', 'purple', 'orange', 'red'])
    ax.set_ylabel('Event count')
    ax.set_title('Channel Activity')

    plt.tight_layout()
    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    fig_path = outdir / 'io_test.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化: {fig_path}")


def live_full_sensory(n_steps: int, device: str):
    """实时全感官感知循环"""
    screen_w, screen_h = get_screen_size()
    print(f"平台: {PLATFORM}")
    print(f"屏幕: {screen_w}x{screen_h}")

    config = get_config()
    model = MultimodalJSpaceModel(config).to(device)

    model_path = Path('outputs/multimodal_model.pt')
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model'], strict=False)
            print(f"已加载模型: {model_path}")
        except Exception as e:
            print(f"模型加载失败，用随机初始化: {e}")
    else:
        print("未找到训练模型，用随机初始化")
    model.eval()

    print(f"\n专家分工: {model.expert_modality}")

    print("\n启动全感官流（摄像头+麦克风+屏幕+键盘+鼠标）...")
    stream = FullSensoryStream(
        use_camera=True, use_mic=True, use_desktop=True, img_size=(32, 32),
    )
    stream.start()

    state = model.init_state(1, torch.device(device))
    step_log = []

    print(f"\n运行 {n_steps} 步感知循环...")
    print("（移动鼠标、按键、对摄像头说话/做动作，模型会持续感知）")
    print("=" * 60)

    try:
        for step in range(n_steps):
            data = stream.get_latest()

            modality = None
            input_tensor = None

            if data['keyboard']:
                key_events = data['keyboard']
                key_ids = []
                for ev in key_events[:8]:
                    k = ev.data if isinstance(ev.data, str) else ' '
                    key_ids.append(ord(k[0]) if k and len(k) == 1 else 0)
                if key_ids:
                    modality = 'keyboard'
                    input_tensor = torch.tensor([key_ids], dtype=torch.long).to(device)

            elif data['mouse']:
                ev = data['mouse'][0]
                x, y = ev.data
                x_norm = x / screen_w
                y_norm = y / screen_h
                click_l = 1.0 if ev.modifiers.get('button') == 'left' else 0.0
                click_r = 1.0 if ev.modifiers.get('button') == 'right' else 0.0
                modality = 'mouse'
                input_tensor = torch.tensor([[x_norm, y_norm, click_l, click_r]],
                                           dtype=torch.float32).to(device)

            elif data['audio']:
                audio = data['audio'].data
                modality = 'audio'
                input_tensor = torch.tensor([audio], dtype=torch.float32).to(device)

            elif data['screen']:
                img = data['screen'].data
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
                modality = 'screen'
                input_tensor = img_tensor

            elif data['camera']:
                img = data['camera'].data
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
                modality = 'image'
                input_tensor = img_tensor

            if modality is None:
                time.sleep(0.2)
                continue

            with torch.no_grad():
                x = model.encode_modality(modality, input_tensor)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if x.dim() == 3:
                    x = x[:, -1, :]
                if x.shape[0] != 1:
                    x = x[-1:]
                state, _ = model.step(state, x)
                w = state['w']
                w_norm = w.norm(dim=-1).mean().item()

            step_log.append({
                'step': step, 'modality': modality, 'w_norm': w_norm,
            })

            if step % 5 == 0:
                if step % 10 == 0 and step > 0:
                    with torch.no_grad():
                        audio_out = model.audio_decoder(w)[0].cpu().numpy()
                    stream.play_audio(audio_out)

                mouse_pos = data.get('mouse_pos', (0, 0))
                key_buf = data.get('keyboard_buffer', '')[:20]
                print(f"  step {step:3d} | mod {modality:8s} | ||w|| {w_norm:.3f} | "
                      f"mouse={mouse_pos} | keys={repr(key_buf)}")

            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        stream.stop()

    # 可视化
    if step_log:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        steps = [s['step'] for s in step_log]
        w_norms = [s['w_norm'] for s in step_log]
        modalities = [s['modality'] for s in step_log]
        color_map = {'image': 'green', 'screen': 'purple', 'audio': 'blue',
                     'keyboard': 'orange', 'mouse': 'red'}
        colors = [color_map.get(m, 'gray') for m in modalities]
        ax.scatter(steps, w_norms, c=colors, alpha=0.7, s=30)
        ax.set_xlabel('Step')
        ax.set_ylabel('||w||')
        ax.set_title('Workspace Norm by Modality')
        ax.grid(True, alpha=0.3)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=m) for m, c in color_map.items()]
        ax.legend(handles=legend_elements)

        ax = axes[1]
        mod_counts = {}
        for m in modalities:
            mod_counts[m] = mod_counts.get(m, 0) + 1
        ax.bar(mod_counts.keys(), mod_counts.values(),
               color=[color_map.get(m, 'gray') for m in mod_counts.keys()])
        ax.set_ylabel('Count')
        ax.set_title('Modality Distribution')

        plt.tight_layout()
        outdir = Path('outputs')
        outdir.mkdir(exist_ok=True)
        fig_path = outdir / 'full_sensory_live.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化: {fig_path}")

    print(f"\n完成，共 {len(step_log)} 步")


def main():
    parser = argparse.ArgumentParser(description='JspaceAI 全感官交互')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'live'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'

    if args.mode == 'test':
        test_io()
    elif args.mode == 'live':
        live_full_sensory(args.steps, device)


if __name__ == '__main__':
    main()
