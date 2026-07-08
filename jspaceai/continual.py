"""
Continual learning utilities shared by interactive runtimes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .language_model import (
    EWCOptimizer,
    ExperienceReplay,
    ExpertPlasticity,
)


class OnlineLanguageLearner:
    """
    Stateful online learner for chat/runtime usage.

    Unlike the old ad hoc SGD update, this keeps replay, EWC regularization,
    and periodic consolidation alive across the whole session.
    """

    def __init__(
        self,
        model,
        config,
        tokenizer,
        device: str = "cpu",
        lr: float = 5e-3,
        ewc_lambda: float = 0.05,
        seq_len: int = 48,
        replay_batch_size: int = 4,
        replay_weight: float = 0.5,
        consolidate_every: int = 20,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.replay_batch_size = replay_batch_size
        self.replay_weight = replay_weight
        self.consolidate_every = consolidate_every

        self.optimizer = EWCOptimizer(model, lr=lr, ewc_lambda=ewc_lambda)
        self.replay_buffer = ExperienceReplay(capacity=500, seq_len=seq_len)
        self.plasticity = ExpertPlasticity(num_experts=config.num_experts)
        self.step_count = 0

    def _prepare_sequence(self, text: str) -> torch.Tensor | None:
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) < 4:
            return None
        if len(token_ids) < self.seq_len:
            token_ids = token_ids * (self.seq_len // len(token_ids) + 1)
        token_ids = token_ids[:self.seq_len]
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    def learn_text(self, text: str) -> dict | None:
        token_seq = self._prepare_sequence(text)
        if token_seq is None:
            return None

        self.model.train()
        logits, info = self.model(token_seq)
        pred = logits[:, :-1]
        target = token_seq[:, 1:]
        loss = F.cross_entropy(
            pred.reshape(-1, self.config.vocab_size),
            target.reshape(-1),
        )

        replay_loss = torch.tensor(0.0, device=self.device)
        replay_seq = self.replay_buffer.sample(self.replay_batch_size)
        if replay_seq is not None:
            replay_seq = replay_seq.to(self.device)
            replay_logits, _ = self.model(replay_seq)
            replay_pred = replay_logits[:, :-1]
            replay_target = replay_seq[:, 1:]
            replay_loss = F.cross_entropy(
                replay_pred.reshape(-1, self.config.vocab_size),
                replay_target.reshape(-1),
            )

        total_task_loss = loss + self.replay_weight * replay_loss
        total_loss = self.optimizer.step(total_task_loss)

        self.plasticity.update(info["alpha"].detach(), token_seq.detach())
        self.replay_buffer.push(token_seq.detach().cpu())
        self.step_count += 1

        if self.step_count % self.consolidate_every == 0:
            self.optimizer.consolidate(token_seq.detach(), n_samples=10)

        self.model.eval()
        return {
            "loss": float(loss.item()),
            "replay_loss": float(replay_loss.item()),
            "total_loss": float(total_loss),
            "w_norm_mean": float(info["w_norm"].mean().item()),
            "usage": self.plasticity.usage.tolist(),
            "step": self.step_count,
        }
