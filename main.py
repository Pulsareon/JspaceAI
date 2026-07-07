#!/usr/bin/env python3
"""
JspaceAI —— 主程序入口

对比实验：
    1. JSpaceModel（工作空间 + J-space 广播 + ODE 动力学）
    2. FlatBaseline（同样参数量的扁平 MLP）

任务：多模态连续时间序列预测（4 种模式切换）

输出：
    - 训练/评估 loss 曲线对比
    - 注意力热力图（哪个专家在哪个时段被激活）
    - 工作空间 ||w|| 演化曲线（输出门控信号）
    - 预测 vs 真实序列对比

运行：
    python main.py
    python main.py --steps 1000 --device cuda
"""
from __future__ import annotations

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from jspaceai import (
    JSpaceConfig, JSpaceModel, FlatBaseline,
    ContinuousSequenceTask, Trainer,
)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(n_steps: int, device: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # 任务
    task = ContinuousSequenceTask(input_dim=8, seq_len=64, seed=42)

    # 模型 1: JSpaceModel
    config = JSpaceConfig(
        input_dim=8,
        workspace_dim=32,
        expert_dim=16,
        num_experts=5,
        num_wells=4,
        ode_steps=4,
        dt=0.1,
        tau_w=0.3,
        output_threshold=0.5,
        jacobian_sparsity=8,
        noise_std=0.01,
    )
    jspace_model = JSpaceModel(config)

    # 模型 2: FlatBaseline（参数量对齐）
    # JSpaceModel 大约的参数量：
    #   专家: 5 × (4×16 + 4 + 16×8 + 16×32 + 16×32) ≈ 5 × 1108 = 5540
    #   工作空间: (8+32)×32 + 32 + 32×32 + 32 ≈ 2144
    #   预测头: 32×32 + 32 + 32×8 + 8 ≈ 1384
    #   总计 ≈ 9000
    flat_model = FlatBaseline(input_dim=8, hidden_dim=90, num_layers=2)

    print("=" * 60)
    print("JspaceAI 对比实验")
    print("=" * 60)
    print(f"任务: 多模态连续序列预测 (input_dim=8, seq_len=64)")
    print(f"训练步数: {n_steps}")
    print(f"设备: {device}")
    print()
    print(f"JSpaceModel 参数量: {count_params(jspace_model):,}")
    print(f"FlatBaseline 参数量: {count_params(flat_model):,}")
    print()

    # 训练
    print("-" * 60)
    print("训练 JSpaceModel...")
    trainer_js = Trainer(jspace_model, task, lr=1e-3, device=device)
    history_js = trainer_js.train(n_steps=n_steps, batch_size=32, eval_interval=n_steps // 10)

    print()
    print("-" * 60)
    print("训练 FlatBaseline...")
    trainer_flat = Trainer(flat_model, task, lr=1e-3, device=device)
    history_flat = trainer_flat.train(n_steps=n_steps, batch_size=32, eval_interval=n_steps // 10)

    # 评估
    print()
    print("=" * 60)
    print("最终评估")
    print("=" * 60)
    eval_xs = task.generate_batch(128)
    eval_loss_js = trainer_js.evaluate(eval_xs)
    eval_loss_flat = trainer_flat.evaluate(eval_xs)
    print(f"JSpaceModel  eval MSE: {eval_loss_js:.6f}")
    print(f"FlatBaseline eval MSE: {eval_loss_flat:.6f}")
    winner = "JSpaceModel" if eval_loss_js < eval_loss_flat else "FlatBaseline"
    improvement = abs(eval_loss_js - eval_loss_flat) / max(eval_loss_js, eval_loss_flat) * 100
    print(f"胜者: {winner} (相对优势 {improvement:.1f}%)")

    # 可视化
    print()
    print("生成可视化...")

    # 1. Loss 曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(history_js, label='JSpaceModel', alpha=0.7, linewidth=0.8)
    ax.plot(history_flat, label='FlatBaseline', alpha=0.7, linewidth=0.8)
    # 平滑曲线
    if len(history_js) > 20:
        smooth_js = np.convolve(history_js, np.ones(20)/20, mode='valid')
        smooth_flat = np.convolve(history_flat, np.ones(20)/20, mode='valid')
        ax.plot(smooth_js, label='JSpace (smoothed)', linewidth=2)
        ax.plot(smooth_flat, label='Flat (smoothed)', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. 注意力热力图
    test_xs = task.generate_batch(1)  # 单条序列
    alpha = trainer_js.get_attention(test_xs)  # (1, T, num_experts)
    if alpha is not None:
        ax = axes[0, 1]
        alpha_np = alpha[0].cpu().numpy()  # (T, num_experts)
        im = ax.imshow(alpha_np.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Expert index')
        ax.set_title('J-space Attention (which expert is active)')
        plt.colorbar(im, ax=ax)
        # 标注时段边界
        for i in range(1, 64 // 16):
            ax.axvline(x=i * 16, color='cyan', linestyle='--', alpha=0.7)
        ax.set_xticks(range(0, 64, 16))

    # 3. 工作空间 ||w|| 演化
    jspace_model.eval()
    with torch.no_grad():
        xs = test_xs.to(device)
        _, info = jspace_model(xs)
    w_norm = info['w_norm'][0].cpu().numpy()

    ax = axes[1, 0]
    ax.plot(w_norm, label='||w||', linewidth=2)
    ax.axhline(y=config.output_threshold, color='r', linestyle='--',
               label=f'threshold={config.output_threshold}', alpha=0.7)
    ax.set_xlabel('Time step')
    ax.set_ylabel('||w||')
    ax.set_title('Workspace norm (output gating signal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for i in range(1, 64 // 16):
        ax.axvline(x=i * 16, color='gray', linestyle='--', alpha=0.5)

    # 4. 预测 vs 真实（取第 0 维）
    ax = axes[1, 1]
    with torch.no_grad():
        xs = test_xs.to(device)
        preds_js, _ = jspace_model(xs)
        preds_flat, _ = flat_model(xs)
    true_seq = test_xs[0, 1:, 0].numpy()
    pred_js = preds_js[0, :-1, 0].cpu().numpy()
    pred_flat = preds_flat[0, :-1, 0].cpu().numpy()

    ax.plot(true_seq, label='True', linewidth=2, color='black')
    ax.plot(pred_js, label='JSpace', alpha=0.8)
    ax.plot(pred_flat, label='Flat', alpha=0.8)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value (dim 0)')
    ax.set_title('Prediction vs True (dim 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for i in range(1, 64 // 16):
        ax.axvline(x=i * 16, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig_path = outdir / 'experiment.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {fig_path}")
    plt.close()

    # 保存模型
    torch.save({
        'jspace_model': jspace_model.state_dict(),
        'flat_model': flat_model.state_dict(),
        'config': config,
        'eval_loss_js': eval_loss_js,
        'eval_loss_flat': eval_loss_flat,
    }, outdir / 'models.pt')

    print()
    print("=" * 60)
    print("实验完成")
    print("=" * 60)
    print(f"JSpaceModel MSE:  {eval_loss_js:.6f}")
    print(f"FlatBaseline MSE: {eval_loss_flat:.6f}")
    print(f"胜者: {winner}")
    print()
    print("关键观察点（看 experiment.png）:")
    print("  1. Loss 曲线: JSpace 是否收敛更快/更低？")
    print("  2. 注意力热力图: 不同时段是否激活不同专家？")
    print("  3. ||w|| 曲线: 模式切换时是否有明显尖峰？")
    print("  4. 预测对比: 模式切换处哪个模型更鲁棒？")


def main():
    parser = argparse.ArgumentParser(description='JspaceAI 对比实验')
    parser.add_argument('--steps', type=int, default=500,
                        help='训练步数 (default: 500)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu / cuda / mps)')
    parser.add_argument('--outdir', type=str, default='outputs',
                        help='输出目录')
    args = parser.parse_args()

    # 自动检测设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    run_experiment(
        n_steps=args.steps,
        device=device,
        outdir=Path(args.outdir),
    )


if __name__ == '__main__':
    main()
