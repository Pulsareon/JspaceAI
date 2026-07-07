#!/usr/bin/env python3
"""
JspaceAI v2 —— 基于 Anthropic J-space 论文优化版

新增功能：
    1. J-lens：观测模型内部"想法"（每个 ODE 子步的 workspace 读出）
    2. Directed Modulation：指令模型"想某概念"，验证 workspace 可被 top-down 调制
    3. Selectivity 验证：ablate workspace，看是否只影响灵活推理
    4. W 轨迹记录：forward 时记录每个子步的 w，用于 J-lens 训练和可视化

运行：
    python main_v2.py
    python main_v2.py --steps 200 --device mps
"""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from jspaceai import (
    LanguageConfig, JSpaceLanguageModel, EvolutionTrainer,
    CharTokenizer, load_shakespeare,
    JLensConfig, JLensSuite,
    WorkspaceAblator, DirectedModulation,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(n_steps: int, device: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 数据
    text = load_shakespeare()
    tokenizer = CharTokenizer.from_text(text)
    print(f"文本: {len(text)} 字符, 词汇表: {tokenizer.vocab_size}")

    # 2. 模型
    config = LanguageConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16, input_dim=8,
        workspace_dim=32, expert_dim=16,
        num_experts=5, num_wells=4,
        ode_steps=4, dt=0.1, tau_w=0.3,
        jacobian_sparsity=8, noise_std=0.005,
    )
    model = JSpaceLanguageModel(config)
    print(f"模型参数: {count_params(model):,}")

    # 3. J-lens 套件
    jlens_config = JLensConfig(
        n_substeps=config.ode_steps,
        workspace_dim=config.workspace_dim,
        vocab_size=config.vocab_size,
    )
    jlens_suite = JLensSuite(jlens_config).to(device)

    # 4. 基础进化训练
    print("\n" + "=" * 70)
    print("阶段 1：基础进化训练")
    print("=" * 70)

    trainer = EvolutionTrainer(model, config, lr=5e-3, ewc_lambda=0.05, device=device)
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    trainer.evolve(
        chunks, tokenizer,
        seq_len=48, batch_size=4,
        consolidate_every=30, generate_every=50,
        max_steps=n_steps, prompt_text="To be",
    )

    # 5. 训练 J-lens
    print("\n" + "=" * 70)
    print("阶段 2：训练 J-lens 探针")
    print("=" * 70)

    jlens_optimizer = torch.optim.Adam(jlens_suite.parameters(), lr=3e-3)
    model.eval()
    all_tokens = tokenizer.encode(text)

    for jlens_epoch in range(60):
        batch_tokens = []
        for _ in range(8):
            start = np.random.randint(0, len(all_tokens) - 64)
            seq = all_tokens[start:start+48]
            batch_tokens.append(seq)
        token_seq = torch.tensor(batch_tokens, dtype=torch.long).to(device)

        with torch.no_grad():
            logits, info = model(token_seq, record_trajectory=True)

        if 'w_trajectory' in info:
            w_traj = info['w_trajectory']  # (batch, T, n_substeps, workspace_dim)
            # target: 每个位置的下一个 token
            targets = token_seq[:, 1:]  # (batch, T-1)

            jlens_optimizer.zero_grad()
            total_loss = 0
            for substep in range(jlens_config.n_substeps):
                # (batch, T-1, workspace_dim) → 预测 targets
                w_sub = w_traj[:, :-1, substep, :]  # (batch, T-1, workspace_dim)
                pred = jlens_suite.probes[substep](w_sub)  # (batch, T-1, vocab)
                loss = F.cross_entropy(
                    pred.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                )
                total_loss += loss

            total_loss.backward()
            jlens_optimizer.step()

            if (jlens_epoch + 1) % 15 == 0:
                print(f"  J-lens epoch {jlens_epoch+1}/60, loss={total_loss.item():.4f}")

    model.train()

    # 6. J-lens 观测
    print("\n" + "=" * 70)
    print("阶段 3：J-lens 观测——模型在想什么")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        prompt = "To be"
        prompt_ids = tokenizer.encode(prompt)
        token_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        _, info = model(token_tensor, record_trajectory=True)

    if 'w_trajectory' in info:
        w_traj = info['w_trajectory'][0]
        T, n_sub, _ = w_traj.shape
        sub_labels = ['sensory', 'workspace', 'workspace', 'motor']

        print(f"\n提示: '{prompt}'")
        for t in range(min(T, 6)):
            char = prompt[t] if t < len(prompt) else '?'
            print(f"  pos {t} ('{char}'):")
            for s in range(n_sub):
                w = w_traj[t, s].unsqueeze(0)
                probe = jlens_suite.probes[s]
                logits_s = probe(w)
                probs = F.softmax(logits_s, dim=-1)
                topk_probs, topk_idx = probs[0].topk(5)
                tokens_list = [tokenizer.idx_to_char.get(i.item(), '?') for i in topk_idx]
                probs_str = [f"{p:.2f}" for p in topk_probs.tolist()]
                pairs = ' '.join(f"{repr(tc)}({pr})" for tc, pr in zip(tokens_list, probs_str))
                label = sub_labels[s] if s < len(sub_labels) else f's{s}'
                print(f"    子步{s} ({label}): {pairs}")

    # 7. Directed Modulation
    print("\n" + "=" * 70)
    print("阶段 4：Directed Modulation")
    print("=" * 70)

    modulation = DirectedModulation(model, jlens_suite)
    test_concepts = ['R', 'd', 'o', ' ']

    mod_results = []
    model.eval()
    for concept_char in test_concepts:
        if concept_char not in tokenizer.char_to_idx:
            continue
        concept_id = tokenizer.char_to_idx[concept_char]

        with torch.no_grad():
            # 正常生成
            prompt_ids = list(tokenizer.encode("To be"))
            state = model.init_state(1, device)
            for tok in prompt_ids:
                state, _, _, _ = model.step(state, torch.tensor([tok], device=device))

            normal_gen = []
            s = state
            last_tok = prompt_ids[-1]
            for _ in range(25):
                s, logits, _, _ = model.step(s, torch.tensor([last_tok], device=device))
                next_tok = logits[0].argmax().item()
                normal_gen.append(next_tok)
                last_tok = next_tok

            # 调制生成
            mod_gen = []
            prompt_ids2 = list(tokenizer.encode("To be"))
            s = model.init_state(1, device)
            for tok in prompt_ids2:
                s, _, _, _ = model.step(s, torch.tensor([tok], device=device))

            last_tok = prompt_ids2[-1]
            for _ in range(25):
                s = modulation.modulate_state(s, concept_id, strength=3.0)
                s, logits, _, _ = model.step(s, torch.tensor([last_tok], device=device))
                next_tok = logits[0].argmax().item()
                mod_gen.append(next_tok)
                last_tok = next_tok

        normal_str = tokenizer.decode(normal_gen)
        mod_str = tokenizer.decode(mod_gen)
        mod_results.append((concept_char, normal_str, mod_str))
        print(f"\n  注入 '{concept_char}':")
        print(f"    正常: {repr(normal_str[:30])}")
        print(f"    调制: {repr(mod_str[:30])}")

    # 8. Selectivity 验证——对比"自动任务"vs"需要 workspace 的任务"
    print("\n" + "=" * 70)
    print("阶段 5：Selectivity 验证")
    print("=" * 70)

    # 任务 A: 简单续写（自动任务，应该 ablate 不影响）
    test_seqs = []
    for _ in range(8):
        start = np.random.randint(0, len(all_tokens) - 64)
        test_seqs.append(all_tokens[start:start+48])
    test_tensor = torch.tensor(test_seqs, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        # 正常 forward
        normal_logits, _ = model(test_tensor)
        normal_pred = normal_logits[:, :-1].argmax(dim=-1)
        targets = test_tensor[:, 1:]
        normal_acc = (normal_pred == targets).float().mean().item()

        # Ablate workspace forward
        ablate_state = model.init_state(test_tensor.shape[0], device)
        ablate_preds = []
        for t in range(test_tensor.shape[1] - 1):
            ablate_state, logits, _, _ = model.step(ablate_state, test_tensor[:, t])
            w = ablate_state['w']
            k = 5  # ablate top-5 J-lens 方向
            for b in range(w.shape[0]):
                probe = jlens_suite.probes[2]
                w_logits = probe(w[b:b+1])
                topk_vals, topk_idx = w_logits[0].topk(k)
                for idx in topk_idx:
                    d = probe.lens.weight[idx]
                    d_norm = d / (d.norm() + 1e-8)
                    w[b] = w[b] - (w[b] @ d_norm) * d_norm
            ablate_state['w'] = w
            ablate_preds.append(logits.argmax(dim=-1))

        ablate_preds = torch.stack(ablate_preds, dim=1)
        ablate_acc = (ablate_preds == targets).float().mean().item()

    # 任务 B: 长程记忆（需要 workspace 持续装载信息）
    # 构造序列：前半段是"key"，后半段需要回忆 key 的特征
    # 简化版：序列 [A, B, C, ..., A, ?] —— 第二次出现 A 后预测下一个
    # 对字符级，我们看"重复字符"任务：序列里某个字符重复出现，模型要"记住"它
    memory_seqs = []
    for _ in range(8):
        # 构造：随机字符 X 出现在位置 0，然后在位置 40 重复
        x = np.random.choice(all_tokens)
        seq = [np.random.choice(all_tokens) for _ in range(48)]
        seq[0] = x
        seq[40] = x  # 40 步后重复
        memory_seqs.append(seq)
    memory_tensor = torch.tensor(memory_seqs, dtype=torch.long).to(device)

    with torch.no_grad():
        # 正常
        mem_logits, _ = model(memory_tensor)
        mem_pred = mem_logits[:, :-1].argmax(dim=-1)
        mem_targets = memory_tensor[:, 1:]
        # 只看位置 40 之后（需要记忆的位置）
        mem_mask = torch.zeros_like(mem_targets, dtype=torch.bool)
        mem_mask[:, 40:] = True  # 位置 40+ 需要"回忆"
        normal_mem_acc = (mem_pred[mem_mask] == mem_targets[mem_mask]).float().mean().item()

        # Ablate
        abl_state = model.init_state(memory_tensor.shape[0], device)
        abl_preds = []
        for t in range(memory_tensor.shape[1] - 1):
            abl_state, logits, _, _ = model.step(abl_state, memory_tensor[:, t])
            w = abl_state['w']
            for b in range(w.shape[0]):
                probe = jlens_suite.probes[2]
                w_logits = probe(w[b:b+1])
                topk_vals, topk_idx = w_logits[0].topk(5)
                for idx in topk_idx:
                    d = probe.lens.weight[idx]
                    d_norm = d / (d.norm() + 1e-8)
                    w[b] = w[b] - (w[b] @ d_norm) * d_norm
            abl_state['w'] = w
            abl_preds.append(logits.argmax(dim=-1))

        abl_preds = torch.stack(abl_preds, dim=1)
        ablate_mem_acc = (abl_preds[mem_mask] == mem_targets[mem_mask]).float().mean().item()

    print(f"\n  任务 A (简单续写 - 自动认知):")
    print(f"    正常: {normal_acc:.3f}  Ablate: {ablate_acc:.3f}  下降: {normal_acc - ablate_acc:.3f}")
    print(f"\n  任务 B (长程记忆 - 需要 workspace):")
    print(f"    正常: {normal_mem_acc:.3f}  Ablate: {ablate_mem_acc:.3f}  下降: {normal_mem_acc - ablate_mem_acc:.3f}")
    print(f"\n  解读: 任务 B 下降应大于任务 A —— workspace 对长程记忆更关键")

    # 9. 可视化
    print("\n" + "=" * 70)
    print("阶段 6：可视化")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    losses = [h['loss'] for h in trainer.history]
    ax.plot(losses, alpha=0.3, linewidth=0.5, color='blue')
    if len(losses) > 10:
        smoothed = np.convolve(losses, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, linewidth=2, color='blue')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Evolution Loss')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    alpha_history = np.array([h['alpha_mean'] for h in trainer.history])
    for i in range(config.num_experts):
        ax.plot(alpha_history[:, i], label=f'Expert {i}', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Usage')
    ax.set_title('Expert Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    w_norms = [h['w_norm_mean'] for h in trainer.history]
    ax.plot(w_norms, linewidth=1.5, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('||w||')
    ax.set_title('Workspace Norm')
    ax.grid(True, alpha=0.3)

    # J-lens 热力图
    ax = axes[1, 0]
    if 'w_trajectory' in info:
        w_traj = info['w_trajectory'][0]
        T, n_sub, _ = w_traj.shape
        sub_labels = ['sensory', 'workspace', 'workspace', 'motor']
        for s in range(min(n_sub, 4)):
            chars_row = []
            for t in range(T):
                w = w_traj[t, s].unsqueeze(0)
                probe = jlens_suite.probes[s]
                logits_s = probe(w)
                top1 = logits_s.argmax().item()
                chars_row.append(tokenizer.idx_to_char.get(top1, '?'))
            chars = ''.join(chars_row)
            label = sub_labels[s] if s < len(sub_labels) else f's{s}'
            ax.text(0.05, 0.95 - s*0.2, f"{label}: {chars}",
                    transform=ax.transAxes, fontsize=9, fontfamily='monospace',
                    verticalalignment='top')
        ax.set_title('J-lens Top-1 (position x substep)')
        ax.axis('off')

    # Modulation 对比
    ax = axes[1, 1]
    ax.axis('off')
    mod_text = "Directed Modulation:\n\n"
    for concept_char, normal_str, mod_str in mod_results[:3]:
        mod_text += f"Inject '{concept_char}':\n"
        mod_text += f"  N: {repr(normal_str[:20])}\n"
        mod_text += f"  M: {repr(mod_str[:20])}\n\n"
    ax.text(0.05, 0.95, mod_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Directed Modulation')

    # Selectivity
    ax = axes[1, 2]
    tasks = ['Auto\n(continuation)', 'Memory\n(long-range)']
    normal_vals = [normal_acc, normal_mem_acc]
    ablate_vals = [ablate_acc, ablate_mem_acc]
    x = np.arange(len(tasks))
    width = 0.35
    ax.bar(x - width/2, normal_vals, width, label='Normal', color='green', alpha=0.7)
    ax.bar(x + width/2, ablate_vals, width, label='Ablated', color='red', alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Selectivity: Workspace Ablation')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    for i, (n, a) in enumerate(zip(normal_vals, ablate_vals)):
        ax.text(i - width/2, n + 0.005, f'{n:.3f}', ha='center', fontsize=8)
        ax.text(i + width/2, a + 0.005, f'{a:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    fig_path = outdir / 'experiment_v2.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"可视化: {fig_path}")
    plt.close()

    torch.save({
        'model': model.state_dict(),
        'jlens': jlens_suite.state_dict(),
        'config': config,
        'tokenizer_chars': tokenizer.chars,
        'selectivity': {
            'auto': {'normal': normal_acc, 'ablate': ablate_acc},
            'memory': {'normal': normal_mem_acc, 'ablate': ablate_mem_acc},
        },
    }, outdir / 'model_v2.pt')

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)
    print(f"基础训练: {len(trainer.history)} 步")
    print(f"J-lens: {jlens_config.n_substeps} 个探针")
    print(f"Selectivity:")
    print(f"  自动任务下降: {normal_acc - ablate_acc:.3f}")
    print(f"  记忆任务下降: {normal_mem_acc - ablate_mem_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description='JspaceAI v2')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    run_experiment(n_steps=args.steps, device=device, outdir=Path(args.outdir))


if __name__ == '__main__':
    main()
