"""
Consensus state utilities.

The workspace vector is still the primary state, but these helpers expose a
small structured summary that other subsystems can share without needing to
decode the full latent every time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class ConsensusSlot:
    index: int
    label: str
    weight: float


@dataclass
class ConsensusSnapshot:
    modality: str
    workspace_norm: float
    confidence: float
    attention_entropy: float
    slots: list[ConsensusSlot]

    @classmethod
    def from_workspace(
        cls,
        w: torch.Tensor,
        alpha: torch.Tensor | None,
        modality: str,
        labels: list[str] | None = None,
        top_k: int = 3,
    ) -> "ConsensusSnapshot":
        workspace_norm = float(w.norm(dim=-1).mean().item())
        slots: list[ConsensusSlot] = []
        attention_entropy = 0.0
        confidence = 0.0

        if alpha is not None and alpha.numel() > 0:
            probs = alpha[0].detach().cpu()
            top_vals, top_idx = probs.topk(min(top_k, probs.numel()))
            slots = [
                ConsensusSlot(
                    index=int(idx.item()),
                    label=labels[int(idx.item())] if labels and int(idx.item()) < len(labels)
                    else f"expert_{idx.item()}",
                    weight=float(val.item()),
                )
                for val, idx in zip(top_vals, top_idx)
            ]

            probs_clamped = probs.clamp_min(1e-8)
            attention_entropy = float((-(probs_clamped * probs_clamped.log()).sum()).item())
            max_entropy = math.log(max(2, probs.numel()))
            confidence = float(max(0.0, 1.0 - attention_entropy / max_entropy))

        return cls(
            modality=modality,
            workspace_norm=workspace_norm,
            confidence=confidence,
            attention_entropy=attention_entropy,
            slots=slots,
        )

    def primary_slot(self) -> ConsensusSlot | None:
        return self.slots[0] if self.slots else None

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "workspace_norm": self.workspace_norm,
            "confidence": self.confidence,
            "attention_entropy": self.attention_entropy,
            "slots": [
                {"index": slot.index, "label": slot.label, "weight": slot.weight}
                for slot in self.slots
            ],
        }
