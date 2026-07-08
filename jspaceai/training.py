"""
Reusable training utilities for the JSpace language model.

The goal is to keep training scalable without hard-wiring it to one script:
sampling, validation, checkpointing, replay, and EWC all live behind a single
session object that can be reused by CLI tools, tests, and future workers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .language_data import CharTokenizer
from .language_model import (
    EWCOptimizer,
    ExperienceReplay,
    ExpertPlasticity,
    JSpaceLanguageModel,
    LanguageConfig,
)


@dataclass
class LanguageTrainingConfig:
    seq_len: int = 64
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    ewc_lambda: float = 0.05
    max_grad_norm: float = 0.5
    replay_capacity: int = 500
    replay_batch_size: int = 4
    replay_weight: float = 0.5
    consolidate_every: int = 50
    consolidate_samples: int = 10
    validate_every: int = 50
    validate_batches: int = 4
    save_every: int = 100
    train_fraction: float = 0.98
    use_euler_during_train: bool = True


@contextmanager
def expert_integration_mode(model, use_rk4: bool):
    """Temporarily set expert integration mode."""
    original = [expert.use_rk4 for expert in model.experts]
    for expert in model.experts:
        expert.use_rk4 = use_rk4
    try:
        yield
    finally:
        for expert, enabled in zip(model.experts, original):
            expert.use_rk4 = enabled


class TokenBatchSampler:
    """Random contiguous sampler over a token stream with a train/val split."""

    def __init__(
        self,
        token_ids: list[int],
        seq_len: int = 64,
        train_fraction: float = 0.98,
    ):
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        self.seq_len = seq_len
        min_len = seq_len + 2
        if len(token_ids) < min_len:
            repeats = min_len // len(token_ids) + 1
            token_ids = (token_ids * repeats)[:min_len]

        split = int(len(token_ids) * train_fraction)
        split = min(max(split, min_len), len(token_ids))
        self.train_tokens = token_ids[:split]
        self.val_tokens = token_ids[split:] if len(token_ids) - split >= min_len else token_ids[:split]

    def sample(self, batch_size: int, split: str = "train", device: str = "cpu") -> torch.Tensor:
        tokens = self.train_tokens if split == "train" else self.val_tokens
        max_start = max(1, len(tokens) - self.seq_len - 1)
        rows = []
        for _ in range(batch_size):
            start = torch.randint(0, max_start, (1,)).item()
            rows.append(tokens[start:start + self.seq_len])
        return torch.tensor(rows, dtype=torch.long, device=device)


def save_language_checkpoint(
    path: str | Path,
    model: JSpaceLanguageModel,
    config: LanguageConfig,
    tokenizer: CharTokenizer,
    trainer_state: dict | None = None,
    metadata: dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "tokenizer_chars": tokenizer.chars,
        "trainer": trainer_state or {},
        "metadata": metadata or {},
    }, path)


class LanguageTrainingSession:
    """Stateful trainer for scalable language-model training."""

    def __init__(
        self,
        model: JSpaceLanguageModel,
        model_config: LanguageConfig,
        tokenizer: CharTokenizer,
        train_config: LanguageTrainingConfig | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.train_config = train_config or LanguageTrainingConfig()
        self.device = device
        self.optimizer = EWCOptimizer(
            self.model,
            lr=self.train_config.lr,
            ewc_lambda=self.train_config.ewc_lambda,
            max_grad_norm=self.train_config.max_grad_norm,
            weight_decay=self.train_config.weight_decay,
        )
        self.replay_buffer = ExperienceReplay(
            capacity=self.train_config.replay_capacity,
            seq_len=self.train_config.seq_len,
        )
        self.plasticity = ExpertPlasticity(num_experts=model_config.num_experts)
        self.global_step = 0
        self.history: list[dict] = []

    def learn_batch(self, token_seq: torch.Tensor) -> dict:
        token_seq = token_seq.to(self.device)
        self.model.train()
        logits, info = self.model(token_seq)
        pred = logits[:, :-1]
        target = token_seq[:, 1:]
        loss = F.cross_entropy(
            pred.reshape(-1, self.model_config.vocab_size),
            target.reshape(-1),
        )

        replay_loss = torch.tensor(0.0, device=self.device)
        replay_seq = self.replay_buffer.sample(self.train_config.replay_batch_size)
        if replay_seq is not None:
            replay_seq = replay_seq.to(self.device)
            replay_logits, _ = self.model(replay_seq)
            replay_pred = replay_logits[:, :-1]
            replay_target = replay_seq[:, 1:]
            replay_loss = F.cross_entropy(
                replay_pred.reshape(-1, self.model_config.vocab_size),
                replay_target.reshape(-1),
            )

        total_task_loss = loss + self.train_config.replay_weight * replay_loss
        total_loss = self.optimizer.step(total_task_loss)

        self.plasticity.update(info["alpha"].detach(), token_seq.detach())
        self.replay_buffer.push(token_seq.detach().cpu())

        return {
            "loss": float(loss.item()),
            "replay_loss": float(replay_loss.item()),
            "total_loss": float(total_loss),
            "w_norm_mean": float(info["w_norm"].mean().item()),
            "expert_usage": self.plasticity.usage.tolist(),
        }

    @torch.no_grad()
    def evaluate(self, sampler: TokenBatchSampler) -> float:
        was_training = self.model.training
        self.model.eval()
        losses = []
        for _ in range(max(1, self.train_config.validate_batches)):
            token_seq = sampler.sample(
                self.train_config.batch_size,
                split="val",
                device=self.device,
            )
            logits, _ = self.model(token_seq)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.model_config.vocab_size),
                token_seq[:, 1:].reshape(-1),
            )
            losses.append(loss.item())
        if was_training:
            self.model.train()
        return float(sum(losses) / len(losses))

    def fit_text(
        self,
        text: str,
        max_steps: int,
        checkpoint_path: str | Path | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        return self.fit_tokens(
            self.tokenizer.encode(text),
            max_steps=max_steps,
            checkpoint_path=checkpoint_path,
            on_progress=on_progress,
        )

    def fit_tokens(
        self,
        token_ids: list[int],
        max_steps: int,
        checkpoint_path: str | Path | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        sampler = TokenBatchSampler(
            token_ids,
            seq_len=self.train_config.seq_len,
            train_fraction=self.train_config.train_fraction,
        )
        use_rk4 = not self.train_config.use_euler_during_train
        with expert_integration_mode(self.model, use_rk4=use_rk4):
            for _ in range(max_steps):
                batch = sampler.sample(
                    self.train_config.batch_size,
                    split="train",
                    device=self.device,
                )
                stats = self.learn_batch(batch)
                self.global_step += 1
                stats["step"] = self.global_step

                if (
                    self.train_config.validate_every > 0
                    and self.global_step % self.train_config.validate_every == 0
                ):
                    stats["val_loss"] = self.evaluate(sampler)

                if (
                    self.train_config.consolidate_every > 0
                    and self.global_step % self.train_config.consolidate_every == 0
                ):
                    self.optimizer.consolidate(
                        batch.detach(),
                        n_samples=self.train_config.consolidate_samples,
                    )

                self.history.append(stats)
                if on_progress:
                    on_progress(stats)

                if (
                    checkpoint_path is not None
                    and self.train_config.save_every > 0
                    and self.global_step % self.train_config.save_every == 0
                ):
                    self.save_checkpoint(checkpoint_path)

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return self.history

    def state_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
            "plasticity": {
                "usage": self.plasticity.usage,
                "expert_specialization": self.plasticity.expert_specialization,
            },
            "train_config": asdict(self.train_config),
            "history": self.history[-200:],
        }

    def load_state_dict(self, state: dict):
        self.global_step = int(state.get("global_step", 0))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        replay = state.get("replay_buffer", [])
        self.replay_buffer.buffer.clear()
        for seq in replay:
            self.replay_buffer.push(seq)
        plasticity = state.get("plasticity", {})
        if "usage" in plasticity:
            self.plasticity.usage = plasticity["usage"].detach().cpu()
        if "expert_specialization" in plasticity:
            self.plasticity.expert_specialization = plasticity["expert_specialization"]
        self.history = list(state.get("history", []))

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None):
        save_language_checkpoint(
            path,
            self.model,
            self.model_config,
            self.tokenizer,
            trainer_state=self.state_dict(),
            metadata=metadata,
        )

    def load_checkpoint_state(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        trainer_state = ckpt.get("trainer")
        if trainer_state:
            self.load_state_dict(trainer_state)
