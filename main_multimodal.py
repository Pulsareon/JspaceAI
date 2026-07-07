#!/usr/bin/env python3
"""
JspaceAI 多模态实时交互 demo

接入摄像头 + 麦克风 + 扬声器，原生支持图像/音频/文本。

模式：
    1. --mode train: 在合成多模态数据上训练（无需摄像头权限）
    2. --mode live: 实时感知-行动循环（需要摄像头/麦克风权限）
    3. --mode eval: 离线评估多模态对齐能力

运行：
    python main_multimodal.py --mode train
    python main_multimodal.py --mode live --steps 100
    python main_multimodal.py --mode eval
"""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from jspaceai import (
    MultimodalConfig, MultimodalJSpaceModel,
    MultimodalStream, SensoryMotorLoop,
)


def get_config(vocab_size: int = 50) -> MultimodalConfig:
    return MultimodalConfig(
        vocab_size=vocab_size,
        embed_dim=16,
        input_dim=8,
        workspace_dim=64,
        expert_dim=24,
        num_experts=8,
        num_wells=4,
        ode_steps=3,
        dt=0.1,
        tau_w=0.3,
        jacobian_sparsity=16,
        noise_std=0.01,
        img_size=32,
        audio_frame_size=1024,
    )


def generate_synthetic_multimodal(batch_size: int = 4, device: str = 'cpu'):
    """生成合成多模态数据：3 个概念对应 3 种模态特征"""
    import cv2
    concepts = ['A', 'B', 'C']
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
    freqs = [200, 800, 400]

    images, audios, tokens = [], [], []
    for _ in range(batch_size):
        idx = np.random.randint(3)
        img = np.zeros((32, 32, 3), dtype=np.float32)
        color = colors[idx]
        if idx == 0:
            cv2.circle(img, (16, 16), 8, color, -1)
        elif idx == 1:
            img[8:24, 8:24] = color
        else:
            for y in range(32):
                w = min(y, 31 - y)
                if 8 < y < 24:
                    img[31-y, 16-w:16+w] = color
        images.append(img.transpose(2, 0, 1))

        t = np.linspace(0, 1024/16000, 1024)
        audio = np.sin(2 * np.pi * freqs[idx] * t).astype(np.float32) * 0.5
        audios.append(audio)
        tokens.append(idx)

    images = torch.tensor(np.stack(images)).to(device) / 255.0 * 2 - 1
    audios = torch.tensor(np.stack(audios)).to(device)
    tokens = torch.tensor(tokens, dtype=torch.long).to(device)
    return images, audios, tokens


def train_multimodal(n_steps: int, device: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    config = get_config()
    model = MultimodalJSpaceModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"专家模态: {model.expert_modality}")
    print(f"\n训练 {n_steps} 步...")
    history = []

    for step in range(n_steps):
        images, audios, tokens = generate_synthetic_multimodal(8, device)
        modality_choice = np.random.randint(3)
        total_loss = 0

        if modality_choice == 0:
            outputs, _ = model.forward_multimodal('image', images)
            text_loss = F.cross_entropy(outputs['text_logits'], tokens)
            audio_loss = F.mse_loss(outputs['audio'], audios)
            total_loss = text_loss + 0.1 * audio_loss
        elif modality_choice == 1:
            outputs, _ = model.forward_multimodal('audio', audios)
            text_loss = F.cross_entropy(outputs['text_logits'], tokens)
            img_loss = F.mse_loss(outputs['image'], images)
            total_loss = text_loss + 0.1 * img_loss
        else:
            outputs, _ = model.forward_multimodal('text', tokens)
            img_loss = F.mse_loss(outputs['image'], images)
            audio_loss = F.mse_loss(outputs['audio'], audios)
            total_loss = 0.1 * img_loss + 0.1 * audio_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(total_loss.item())

        if (step + 1) % 20 == 0:
            print(f"  step {step+1:4d} | loss {total_loss.item():.4f} | "
                  f"modality {['img','aud','txt'][modality_choice]}")

    print("\n" + "=" * 60)
    print("跨模态对齐评估")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for mod_name, mod_data in [('image', images), ('audio', audios), ('text', tokens)]:
            outputs, _ = model.forward_multimodal(mod_name, mod_data[:3])
            pred_tokens = outputs['text_logits'].argmax(dim=-1)
            print(f"  输入 {mod_name:6s} → 预测 token: {pred_tokens.cpu().tolist()} "
                  f"(真实: {tokens[:3].cpu().tolist()})")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(history, alpha=0.5, linewidth=0.5)
    if len(history) > 10:
        smoothed = np.convolve(history, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, linewidth=2)
    ax.set_title('Training Loss')
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    with torch.no_grad():
        test_img = images[:1]
        outputs, _ = model.forward_multimodal('image', test_img)

    ax = axes[0, 1]
    ax.imshow(((test_img[0].cpu().permute(1,2,0) + 1) / 2).numpy())
    ax.set_title('Input Image')
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(((outputs['image'][0].cpu().permute(1,2,0) + 1) / 2).clamp(0,1).numpy())
    ax.set_title('Reconstructed Image')
    ax.axis('off')

    ax = axes[1, 0]
    input_audio = audios[0].cpu().numpy()
    ax.plot(input_audio[:200], alpha=0.7, label='input')
    out_audio = outputs['audio'][0].cpu().numpy()
    ax.plot(out_audio[:200], alpha=0.7, label='reconstructed')
    ax.set_title('Audio Waveform')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    logits = outputs['text_logits'][0].cpu().numpy()
    probs = np.exp(logits) / np.exp(logits).sum()
    ax.bar(['A', 'B', 'C'], probs[:3])
    ax.set_title('Text Logits (softmax)')

    ax = axes[1, 2]
    w = outputs['w'][0].cpu().numpy()
    ax.bar(range(len(w)), w)
    ax.set_title('Workspace w (64-dim)')

    plt.tight_layout()
    fig_path = outdir / 'multimodal_train.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化: {fig_path}")

    torch.save({'model': model.state_dict(), 'config': config},
               outdir / 'multimodal_model.pt')
    print(f"模型: {outdir / 'multimodal_model.pt'}")


def live_demo(n_steps: int, device: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    config = get_config()
    model = MultimodalJSpaceModel(config).to(device)

    model_path = outdir / 'multimodal_model.pt'
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"已加载模型: {model_path}")
    else:
        print("未找到训练好的模型，用随机初始化运行")
    model.eval()

    print("\n启动实时多模态流...")
    print("（macOS 会请求摄像头和麦克风权限，请允许）")
    stream = MultimodalStream(
        use_camera=True, use_mic=True,
        img_size=(32, 32), sample_rate=16000, audio_frame_size=1024,
    )
    loop = SensoryMotorLoop(model, stream, device=device)
    step_log = []

    def on_step(info):
        mod = info.get('modality', '?')
        step = info.get('step', 0)
        w_norm = info.get('w_norm', torch.tensor([0])).mean().item()
        step_log.append({'step': step, 'modality': mod, 'w_norm': w_norm})
        if step % 5 == 0:
            print(f"  step {step:3d} | modality {mod:5s} | ||w|| {w_norm:.3f}")

    loop.run(n_steps=n_steps, interval=0.2, on_step=on_step)

    if step_log:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        steps = [s['step'] for s in step_log]
        w_norms = [s['w_norm'] for s in step_log]
        modalities = [s['modality'] for s in step_log]
        colors = ['blue' if m == 'audio' else 'green' for m in modalities]
        ax.scatter(steps, w_norms, c=colors, alpha=0.6, s=20)
        ax.set_xlabel('Step')
        ax.set_ylabel('||w||')
        ax.set_title('Workspace Norm During Live (blue=audio, green=image)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = outdir / 'multimodal_live.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"可视化: {fig_path}")


def eval_multimodal(device: str, outdir: Path):
    model_path = outdir / 'multimodal_model.pt'
    if not model_path.exists():
        print("请先运行 --mode train")
        return

    config = get_config()
    model = MultimodalJSpaceModel(config).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print("多模态对齐评估")
    print("=" * 60)

    images, audios, tokens = generate_synthetic_multimodal(6, device)

    with torch.no_grad():
        for mod_name, mod_data in [('image', images), ('audio', audios), ('text', tokens)]:
            outputs, _ = model.forward_multimodal(mod_name, mod_data[:3])
            pred_tokens = outputs['text_logits'].argmax(dim=-1)
            correct = (pred_tokens == tokens[:3]).float().mean().item()
            print(f"  输入 {mod_name:6s} → token 准确率: {correct:.3f}")


def main():
    parser = argparse.ArgumentParser(description='JspaceAI 多模态')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'live', 'eval'])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'

    if args.mode == 'train':
        train_multimodal(args.steps, device, Path(args.outdir))
    elif args.mode == 'live':
        live_demo(args.steps, device, Path(args.outdir))
    elif args.mode == 'eval':
        eval_multimodal(device, Path(args.outdir))


if __name__ == '__main__':
    main()
