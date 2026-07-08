"""
Serializable event types for the workspace-centered runtime.

These data packets are intentionally small and plain. They give local modules,
future worker processes, and external transports the same language for passing
workspace state around without sharing live Python objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import torch

from .consensus import ConsensusSnapshot


def _workspace_vector(w: torch.Tensor | np.ndarray | list[float]) -> list[float]:
    if isinstance(w, torch.Tensor):
        data = w.detach().cpu()
        if data.dim() > 1:
            data = data[0]
        return data.reshape(-1).tolist()
    if isinstance(w, np.ndarray):
        data = w
        if data.ndim > 1:
            data = data[0]
        return data.reshape(-1).astype(float).tolist()
    return [float(v) for v in w]


def _consensus_dict(consensus: ConsensusSnapshot | dict | None) -> dict | None:
    if consensus is None:
        return None
    if isinstance(consensus, ConsensusSnapshot):
        return consensus.to_dict()
    return consensus


@dataclass
class WorkspaceEvent:
    """A transport-friendly snapshot of workspace state."""

    step: int
    modality: str
    workspace: list[float]
    consensus: dict | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    kind: str = "workspace"

    @classmethod
    def from_tensor(
        cls,
        w: torch.Tensor | np.ndarray | list[float],
        modality: str,
        step: int = 0,
        consensus: ConsensusSnapshot | dict | None = None,
        context: dict | None = None,
    ) -> "WorkspaceEvent":
        return cls(
            step=step,
            modality=modality,
            workspace=_workspace_vector(w),
            consensus=_consensus_dict(consensus),
            context=context or {},
        )

    def workspace_array(self) -> np.ndarray:
        return np.asarray(self.workspace, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "step": self.step,
            "modality": self.modality,
            "workspace": list(self.workspace),
            "consensus": self.consensus,
            "context": self.context,
            "timestamp": self.timestamp,
        }


@dataclass
class ActionEvent:
    """A transport-friendly action decision/result."""

    step: int
    action_idx: int
    action_name: str
    action_params: list[float]
    executed: bool
    risk: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    kind: str = "action"

    @classmethod
    def from_action_info(
        cls,
        action_info: dict,
        step: int = 0,
        context: dict | None = None,
    ) -> "ActionEvent":
        return cls(
            step=step,
            action_idx=int(action_info.get("action_idx", 0)),
            action_name=str(action_info.get("action_name", "observe")),
            action_params=[float(v) for v in action_info.get("action_params", [])],
            executed=bool(action_info.get("executed", False)),
            risk=float(action_info.get("risk", 0.0)),
            context=context or {},
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "step": self.step,
            "action_idx": self.action_idx,
            "action_name": self.action_name,
            "action_params": list(self.action_params),
            "executed": self.executed,
            "risk": self.risk,
            "context": self.context,
            "timestamp": self.timestamp,
        }
