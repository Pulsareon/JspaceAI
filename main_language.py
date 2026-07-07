#!/usr/bin/env python3
"""
JspaceAI 语言版 —— 自主进化实验

模型在 Shakespeare 文本上持续学习，边推理边进化。
观察：
    1. loss 持续下降（在学习）
    2. 生成文本从乱码逐渐变成类 Shakespeare 风格
    3. 专家分工涌现（不同专家处理不同字符模式）
    4. EWC + 经验回放防止灾难性遗忘

运行：
    python main_language.py
    python main_language.py --steps 300 --device mps
"""
from __future__ import annotations

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from jspaceai import (
    LanguageConfig, JSpaceLanguageModel, EvolutionTrainer,
    CharTokenizer, load_shakespeare,
)


def run_language_experiment(n_steps: int, device: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 数据
    text = load_shakespeare()
    tokenizer = CharTokenizer.from_text(text)
    print(f"文本长度: {len(text)} 字符")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"词汇表: {''.join(tokenizer.chars[:50])}...")

    # 切成多段，模拟持续到来的文本流
    chunk_size = 200
    text_stream = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"文本流: {len(text_stream)} 段, 每段 {chunk_size} 字符")

    # 2. 模型
    config = LanguageConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        input_dim=8,           # J-space 输入维度
        workspace_dim=32,      # 工作空间维度
        expert_dim=16,         # 每个专家内部状态
        num_experts=5,         # 5 个并行专家
        num_wells=4,           # 每个专家 4 个吸引子
        ode_steps=3,           # ODE 积分子步
        dt=0.1,
        tau_w=0.3,
        jacobian_sparsity=8,
        noise_std=0.005,
    )
    model = JSpaceLanguageModel(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {n_params:,}")

    # 3. 自主进化训练器
    trainer = EvolutionTrainer(
        model, config,
        lr=5e-3,
        ewc_lambda=0.05,   # 较小的 EWC 权重，让模型能学新东西
        device=device,
    )

    # 4. 进化
    print("\n" + "=" * 70)
    print("自主进化开始")
    print("=" * 70)

    # 限制步数
    history = trainer.evolve(
        text_stream,
        tokenizer,
        seq_len=48,
        batch_size=4,
        consolidate_every=30,
        generate_every=30,
        max_steps=n_steps,
        prompt_text="To be",
    )

    # 5. 总结
    summary = trainer.get_evolution_summary()
    print("\n" + "=" * 70)
    print("进化总结")
    print("=" * 70)
    print(f"总步数: {summary['steps']}")
    print(f"初始 loss: {summary['initial_loss']:.4f}")
    print(f"最终 loss: {summary['final_loss']:.4f}")
    print(f"最低 loss: {summary['min_loss']:.4f}")
    print(f"loss 下降: {(1 - summary['final_loss']/summary['initial_loss'])*100:.1f}%")
    print(f"\n专家最终使用率: {[f'{u:.3f}' for u in summary['expert_usage']]}")

    # 专家专业化（每个专家最常处理的 top-5 字符）
    print("\n专家专业化（top-5 字符）:")
    for i, spec in enumerate(summary['expert_specialization']):
        chars = tokenizer.decode([s[0] for s in spec])
        weights = [f"{s[1]:.1f}" for s in spec]
        display = ' '.join(f"{repr(c)}({w})" for c, w in zip(chars, weights))
        print(f"  专家 {i}: {display}")

    # 6. 最终生成对比
    print("\n" + "=" * 70)
    print("最终生成对比")
    print("=" * 70)
    for prompt in ["To be", "Romeo", "The "]:
        prompt_ids = tokenizer.encode(prompt)
        generated = model.generate(prompt_ids, n_new=100, temperature=0.7, top_k=5)
        sample = prompt + tokenizer.decode(generated)
        print(f"\n提示 '{prompt}':")
        print(f"  {sample}")

    # 7. 可视化
    print("\n生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss 曲线
    ax = axes[0, 0]
    losses = [h['loss'] for h in history]
    ax.plot(losses, alpha=0.3, linewidth=0.5, color='blue', label='raw')
    if len(losses) > 10:
        smoothed = np.convolve(losses, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, linewidth=2, color='blue', label='smoothed')
    ax.set_xlabel('Evolution step')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('Loss during self-evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 专家使用率演化
    ax = axes[0, 1]
    alpha_history = np.array([h['alpha_mean'] for h in history])
    for i in range(config.num_experts):
        ax.plot(alpha_history[:, i], label=f'Expert {i}', alpha=0.8)
    ax.set_xlabel('Evolution step')
    ax.set_ylabel('Attention weight (mean)')
    ax.set_title('Expert usage during evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ||w|| 演化
    ax = axes[1, 0]
    w_norms = [h['w_norm_mean'] for h in history]
    ax.plot(w_norms, linewidth=1.5, color='green')
    ax.set_xlabel('Evolution step')
    ax.set_ylabel('||w|| (mean)')
    ax.set_title('Workspace norm during evolution')
    ax.grid(True, alpha=0.3)

    # 生成样本随时间演化
    ax = axes[1, 1]
    ax.axis('off')
    samples = [(h['step'], h.get('sample', '')) for h in history if 'sample' in h]
    text_lines = []
    for step, sample in samples[:8]:
        s = sample[:60].replace('\n', ' ')
        text_lines.append(f"step {step:3d}: {s}")
    ax.text(0.05, 0.95, '\n'.join(text_lines),
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Generated samples during evolution')

    plt.tight_layout()
    fig_path = outdir / 'language_evolution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {fig_path}")
    plt.close()

    # 保存模型
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'tokenizer_chars': tokenizer.chars,
        'history': history,
        'summary': summary,
    }, outdir / 'language_model.pt')
    print(f"模型已保存: {outdir / 'language_model.pt'}")

    print("\n" + "=" * 70)
    print("实验完成")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='JspaceAI 语言版自主进化实验')
    parser.add_argument('--steps', type=int, default=100,
                        help='文本流段数 (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu / cuda / mps / auto)')
    parser.add_argument('--outdir', type=str, default='outputs',
                        help='输出目录')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    run_language_experiment(
        n_steps=args.steps,
        device=device,
        outdir=Path(args.outdir),
    )


if __name__ == '__main__':
    main()
