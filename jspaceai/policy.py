"""
Action policy utilities centered on the workspace state.

This module keeps action selection, motor parameterization, and gating together
so the embodied runtime can stay focused on sensing and effectors.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_LABELS = {
    0: "observe",
    1: "mouse_move",
    2: "left_click",
    3: "right_click",
    4: "scroll",
}


def compose_action_params(action_idx: int, raw_params: np.ndarray) -> np.ndarray:
    """Map a discrete action choice onto concrete actuator parameters."""
    action = np.zeros(5, dtype=np.float32)
    if action_idx == 1:
        action[0] = float(raw_params[0])
        action[1] = float(raw_params[1])
    elif action_idx == 2:
        action[2] = 1.0 if raw_params[2] >= 0 else 0.0
    elif action_idx == 3:
        action[3] = 1.0 if raw_params[3] >= 0 else 0.0
    elif action_idx == 4:
        action[4] = float(raw_params[4])
    return action


@dataclass
class ReflexRule:
    """Fast path rule that can inhibit or redirect behavior before planning."""

    trigger: str
    condition: Callable
    action: Callable
    priority: int = 0


class MotorController(nn.Module):
    """
    Refines workspace intent into continuous motor parameters.
    """

    def __init__(self, workspace_dim: int, action_dim: int = 5):
        super().__init__()
        self.inverse_model = nn.Sequential(
            nn.Linear(workspace_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(action_dim + workspace_dim, 64),
            nn.ReLU(),
            nn.Linear(64, workspace_dim),
        )
        self.action_dim = action_dim

    def compute_action(self, w: torch.Tensor) -> torch.Tensor:
        return self.inverse_model(w)

    def predict_next(self, w: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward_model(torch.cat([action, w], dim=-1))

    def compute_motor_error(
        self,
        w: torch.Tensor,
        action: torch.Tensor,
        w_actual_next: torch.Tensor,
    ) -> torch.Tensor:
        w_pred = self.predict_next(w, action)
        return F.mse_loss(w_pred, w_actual_next)


class ActionGate:
    """
    Lightweight action gate with reflex handling and execution thresholds.
    """

    def __init__(self, base_threshold: float = 0.3):
        self.base_threshold = base_threshold
        self.reflexes: list[ReflexRule] = []
        self.action_history: deque = deque(maxlen=100)
        self.inhibit_score: float = 0.0

    def add_reflex(self, reflex: ReflexRule):
        self.reflexes.append(reflex)
        self.reflexes.sort(key=lambda r: -r.priority)

    def check_reflexes(self, sensory_state: dict) -> Optional[Callable]:
        for reflex in self.reflexes:
            try:
                if reflex.condition(sensory_state):
                    return reflex.action
            except Exception:
                continue
        return None

    def should_execute(self, action_strength: float, risk: float = 0.0) -> bool:
        threshold = self.base_threshold + risk * 0.5 + self.inhibit_score
        return action_strength > threshold

    def record_action(self, action: np.ndarray, modality: str):
        self.action_history.append({
            "time": time.time(),
            "action": action.tolist() if hasattr(action, "tolist") else action,
            "modality": modality,
        })


class ActionValueModel:
    """
    Linear value function over workspace state for discrete action choice.
    """

    def __init__(self, workspace_dim: int = 64, n_actions: int = 5, learning_rate: float = 0.01):
        self.workspace_dim = workspace_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.action_weights = np.zeros((n_actions, workspace_dim))
        self.habit_counts = np.zeros(n_actions)
        self.action_labels = [ACTION_LABELS.get(i, f"action_{i}") for i in range(n_actions)]

    def compute_values(self, w: np.ndarray) -> np.ndarray:
        return self.action_weights @ w

    def select_action(self, w: np.ndarray, exploration: float = 0.1) -> int:
        values = self.compute_values(w)
        if np.random.random() < exploration:
            return np.random.randint(self.n_actions)
        return int(np.argmax(values))

    def update(self, w: np.ndarray, action: int, reward: float):
        values = self.compute_values(w)
        td_error = reward - values[action]
        self.action_weights[action] += self.lr * td_error * w
        self.habit_counts[action] += 1

    def is_habitual(self, action: int, threshold: int = 10) -> bool:
        return self.habit_counts[action] >= threshold


@dataclass
class ActionDecision:
    action_idx: int
    action_name: str
    action_params: np.ndarray
    raw_action_params: np.ndarray
    action_strength: float
    risk: float
    should_execute: bool

    def to_dict(self, executed: bool) -> dict:
        return {
            "action_idx": self.action_idx,
            "action_name": self.action_name,
            "action_params": self.action_params.tolist(),
            "raw_action_params": self.raw_action_params.tolist(),
            "executed": executed,
            "action_strength": self.action_strength,
            "risk": self.risk,
        }


class ActionPolicy:
    """
    Unified policy around discrete action choice, motor refinement, and gating.
    """

    def __init__(
        self,
        workspace_dim: int,
        action_dim: int = 5,
        n_actions: int = 5,
        learning_rate: float = 0.01,
        exploration: float = 0.2,
        base_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.exploration = exploration
        self.value_model = ActionValueModel(
            workspace_dim=workspace_dim,
            n_actions=n_actions,
            learning_rate=learning_rate,
        )
        self.motor_controller = MotorController(
            workspace_dim=workspace_dim,
            action_dim=action_dim,
        ).to(device)
        self.gate = ActionGate(base_threshold=base_threshold)

    @property
    def action_labels(self) -> list[str]:
        return self.value_model.action_labels

    def add_reflex(self, reflex: ReflexRule):
        self.gate.add_reflex(reflex)

    def check_reflexes(self, sensory_state: dict) -> Optional[Callable]:
        return self.gate.check_reflexes(sensory_state)

    def record_action(self, action: np.ndarray, modality: str):
        self.gate.record_action(action, modality)

    def decide(self, w: torch.Tensor) -> ActionDecision:
        w_np = w[0].detach().cpu().numpy()
        action_idx = self.value_model.select_action(w_np, exploration=self.exploration)
        action_name = self.value_model.action_labels[action_idx]

        with torch.no_grad():
            raw_action = self.motor_controller.compute_action(w)[0].detach().cpu().numpy()
        action_params = compose_action_params(action_idx, raw_action)

        action_strength = float(np.abs(action_params).max())
        risk = 0.5 if action_params[2] > 0.5 or action_params[3] > 0.5 else 0.0
        should_execute = action_idx != 0 and self.gate.should_execute(action_strength, risk)

        return ActionDecision(
            action_idx=action_idx,
            action_name=action_name,
            action_params=action_params,
            raw_action_params=raw_action,
            action_strength=action_strength,
            risk=float(risk),
            should_execute=should_execute,
        )

    def learn(self, w: torch.Tensor, action_idx: int, reward: float = 0.0):
        w_np = w[0].detach().cpu().numpy()
        self.value_model.update(w_np, action_idx, reward)


# Backward-compatible aliases for the old brain-region names.
ReflexArc = ReflexRule
CentralNervousSystem = ActionGate
BasalGanglia = ActionValueModel
Cerebellum = MotorController
