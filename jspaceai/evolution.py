"""
自主进化训练器

核心：模型在推理的同时持续学习。每处理一段文本，参数就更新一次。

进化循环：
    1. 喂入新文本片段
    2. forward + 计算 next-token loss
    3. EWC 优化器更新参数（保护旧知识）
    4. 经验回放：当前片段存入 buffer，定期回放旧片段
    5. 周期性 consolidate EWC（更新 Fisher 信息和锚点）
    6. 追踪专家可塑性统计
    7. 定期生成样本，观察进化效果

这个循环可以无限运行——模型永远不会"训练完成"，它一直在进化。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Callable
from .language_model import (
    JSpaceLanguageModel, LanguageConfig,
    ExperienceReplay, EWCOptimizer, ExpertPlasticity,
)


class EvolutionTrainer:
    """自主进化训练器"""

    def __init__(self, model: JSpaceLanguageModel, config: LanguageConfig,
                 lr: float = 1e-3, ewc_lambda: float = 0.1,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 三大自主进化机制
        self.ewc_optimizer = EWCOptimizer(model, lr=lr, ewc_lambda=ewc_lambda)
        self.replay_buffer = ExperienceReplay(capacity=500, seq_len=64)
        self.plasticity = ExpertPlasticity(num_experts=config.num_experts)

        # 进化历史追踪
        self.history: list[dict] = []

    def learn_step(self, token_seq: torch.Tensor) -> dict:
        """单步学习

        Args:
            token_seq: (batch, T) token indices

        Returns:
            stats: 包含 loss、注意力、专家统计等
        """
        token_seq = token_seq.to(self.device)

        # 1. Forward
        logits, info = self.model(token_seq)

        # 2. Next-token prediction loss
        # logits[:, t] 预测 token_seq[:, t+1]
        pred_logits = logits[:, :-1]  # (batch, T-1, vocab)
        targets = token_seq[:, 1:]    # (batch, T-1)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
        )

        # 3. 经验回放：如果有足够样本，混入旧数据
        replay_loss = torch.tensor(0.0, device=self.device)
        if len(self.replay_buffer.buffer) >= 8:
            replay_seq = self.replay_buffer.sample(4)
            if replay_seq is not None:
                replay_seq = replay_seq.to(self.device)
                replay_logits, _ = self.model(replay_seq)
                replay_pred = replay_logits[:, :-1]
                replay_targets = replay_seq[:, 1:]
                replay_loss = F.cross_entropy(
                    replay_pred.reshape(-1, self.config.vocab_size),
                    replay_targets.reshape(-1),
                )

        # 4. 总 loss + EWC 正则 + 经验回放
        total_task_loss = loss + 0.5 * replay_loss
        total_loss = self.ewc_optimizer.step(total_task_loss)

        # 5. 更新专家可塑性统计
        self.plasticity.update(info['alpha'].detach(), token_seq.detach())

        # 6. 存入经验回放
        self.replay_buffer.push(token_seq.detach().cpu())

        stats = {
            'loss': loss.item(),
            'replay_loss': replay_loss.item() if isinstance(replay_loss, torch.Tensor) else replay_loss,
            'total_loss': total_loss,
            'w_norm_mean': info['w_norm'].mean().item(),
            'alpha_mean': info['alpha'].mean(dim=(0, 1)).detach().cpu().tolist(),
        }
        return stats

    def consolidate(self, data_sample: torch.Tensor):
        """周期性 consolidate EWC——更新参数重要性锚点

        在学完一段文本后调用，把当前知识"固化"
        """
        self.ewc_optimizer.consolidate(data_sample.to(self.device), n_samples=20)

    def evolve(self, text_stream: list[str], tokenizer,
               seq_len: int = 64, batch_size: int = 4,
               consolidate_every: int = 20,
               generate_every: int = 50,
               max_steps: int | None = None,
               prompt_text: str = "To be",
               on_progress: Callable | None = None) -> list[dict]:
        """
        持续进化主循环

        Args:
            text_stream: 文本片段列表（模拟持续到来的数据流）
            tokenizer: CharTokenizer
            seq_len: 序列长度
            batch_size: 每次喂入的 batch 大小
            consolidate_every: 每隔多少步 consolidate EWC
            generate_every: 每隔多少步生成样本观察
            prompt_text: 生成样本的提示词
            on_progress: 回调函数，返回当前进度

        Returns:
            history: 进化历史
        """
        step = 0
        all_tokens = []

        # 把所有文本编码成 token 流
        for text in text_stream:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        # 用滑动窗口在完整 token 流上采样 batch
        # 每个 batch 包含 batch_size 条序列，每条长 seq_len
        # 相邻 batch 之间步进 stride 个序列
        stride = batch_size  # 每个 batch 用 batch_size 个新起点
        n_possible_starts = max(0, len(all_tokens) - seq_len - 1)
        n_batches = max(0, n_possible_starts // stride)

        print(f"自主进化开始：{len(all_tokens)} tokens, {len(text_stream)} 段文本")
        print(f"配置: seq_len={seq_len}, batch_size={batch_size}, "
              f"stride={stride}, n_batches={n_batches}")
        print("=" * 70)

        for batch_idx in range(n_batches):
            # 取一个 batch 的序列（滑动窗口）
            batch_tokens = []
            for i in range(batch_size):
                start = batch_idx * stride + i
                if start + seq_len >= len(all_tokens):
                    batch_tokens.append(all_tokens[-seq_len:])
                else:
                    batch_tokens.append(all_tokens[start:start + seq_len])

            if any(len(b) < seq_len for b in batch_tokens):
                continue

            token_seq = torch.tensor(batch_tokens, dtype=torch.long)
            stats = self.learn_step(token_seq)
            stats['step'] = step
            stats['batch_idx'] = batch_idx
            self.history.append(stats)

            # 周期性 consolidate
            if step > 0 and step % consolidate_every == 0:
                self.consolidate(token_seq)

            # 周期性生成 + 报告
            if step % generate_every == 0 or step == n_batches - 1:
                prompt_ids = tokenizer.encode(prompt_text)
                generated = self.model.generate(
                    prompt_ids, n_new=80, temperature=0.8, top_k=5
                )
                sample = prompt_text + tokenizer.decode(generated)
                stats['sample'] = sample

                print(f"\n[step {step:4d}] loss={stats['loss']:.4f} "
                      f"replay={stats['replay_loss']:.4f} "
                      f"||w||={stats['w_norm_mean']:.3f}")
                print(f"  专家使用率: {[f'{u:.2f}' for u in self.plasticity.usage.tolist()]}")
                print(f"  生成样本: {repr(sample[:120])}...")

                if on_progress:
                    on_progress(stats)

            step += 1

            if max_steps is not None and step >= max_steps:
                break

        print("\n" + "=" * 70)
        print("自主进化完成")
        return self.history

    def get_evolution_summary(self) -> dict:
        """获取进化总结"""
        if not self.history:
            return {}

        losses = [h['loss'] for h in self.history]
        return {
            'steps': len(self.history),
            'final_loss': losses[-1],
            'initial_loss': losses[0],
            'min_loss': min(losses),
            'expert_usage': self.plasticity.usage.tolist(),
            'expert_specialization': self.plasticity.get_stats()['top_specialization'],
            'samples': [h.get('sample', '') for h in self.history if 'sample' in h],
        }
