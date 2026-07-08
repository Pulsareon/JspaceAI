"""
Memory stores for workspace events.

The default store is local and in-memory, but it exposes a small backend
interface that can later be replaced by a vector database or remote service.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np

from .events import WorkspaceEvent


@dataclass
class MemoryRecord:
    event: WorkspaceEvent
    similarity: float | None = None

    @property
    def w(self) -> np.ndarray:
        return self.event.workspace_array()

    @property
    def context(self) -> dict:
        return self.event.context

    @property
    def timestamp(self) -> float:
        return self.event.timestamp

    def to_dict(self) -> dict:
        return {
            "w": self.w,
            "context": self.context,
            "timestamp": self.timestamp,
            "similarity": self.similarity,
            "event": self.event.to_dict(),
        }


class MemoryStore(Protocol):
    def put(self, event: WorkspaceEvent) -> None:
        ...

    def query(self, event_or_workspace, top_k: int = 3) -> list[MemoryRecord]:
        ...

    def size(self) -> int:
        ...


class InMemoryVectorMemoryStore:
    """Small cosine-similarity memory store for workspace events."""

    def __init__(self, capacity: int = 1000, workspace_dim: int = 64):
        self.capacity = capacity
        self.workspace_dim = workspace_dim
        self.records: deque[MemoryRecord] = deque(maxlen=capacity)

    @property
    def memories(self):
        return self.records

    def put(self, event: WorkspaceEvent) -> None:
        self.records.append(MemoryRecord(event=event))

    def query(self, event_or_workspace, top_k: int = 3) -> list[MemoryRecord]:
        if not self.records:
            return []
        query = self._as_workspace(event_or_workspace)
        similarities = [
            float(np.dot(query, record.w) / (np.linalg.norm(query) * np.linalg.norm(record.w) + 1e-8))
            for record in self.records
        ]
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        return [
            MemoryRecord(event=self.records[int(idx)].event, similarity=similarities[int(idx)])
            for idx in top_idx
        ]

    def size(self) -> int:
        return len(self.records)

    def store(self, w: np.ndarray, context: dict | None = None):
        """Backward-compatible write API."""
        event = WorkspaceEvent(
            step=int((context or {}).get("step", 0)),
            modality=str((context or {}).get("modality", "unknown")),
            workspace=self._as_workspace(w).astype(float).tolist(),
            consensus=(context or {}).get("consensus"),
            context=context or {},
            timestamp=time.time(),
        )
        self.put(event)

    def recall(self, w_query: np.ndarray, top_k: int = 3) -> list[dict]:
        """Backward-compatible query API."""
        return [record.to_dict() for record in self.query(w_query, top_k)]

    def _as_workspace(self, event_or_workspace) -> np.ndarray:
        if isinstance(event_or_workspace, WorkspaceEvent):
            return event_or_workspace.workspace_array()
        data = np.asarray(event_or_workspace, dtype=np.float32)
        if data.ndim > 1:
            data = data[0]
        return data.reshape(-1)


# Backward-compatible brain-region name.
Hippocampus = InMemoryVectorMemoryStore
