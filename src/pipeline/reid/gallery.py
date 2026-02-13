from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Iterable
import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / (n + eps)
    if x.ndim == 2:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + eps)
    raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")


@dataclass(frozen=True)
class MatchResult:
    person_id: int
    similarity: float
    created_new: bool = False


class ReIDGallery:
    def __init__(
        self,
        sim_threshold: float = 0.55,
        ema: float = 0.8,
        update_threshold: Optional[float] = None,
    ):
        self.sim_threshold = float(sim_threshold)
        self.update_threshold = float(update_threshold) if update_threshold is not None else float(sim_threshold)
        self.ema = float(ema)

        self._next_id = 1
        self._prototypes: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._prototypes)

    def _create_new(self, emb: np.ndarray, similarity_hint: float = 0.0) -> MatchResult:
        pid = self._next_id
        self._next_id += 1
        self._prototypes[pid] = l2_normalize(emb).copy()
        return MatchResult(pid, float(similarity_hint), created_new=True)

    def add(self, emb: np.ndarray) -> int:
        return self._create_new(emb).person_id

    def match(
        self,
        emb: np.ndarray,
        forbidden_ids: Optional[Iterable[int]] = None,
        *,
        create_new: bool = True,
        min_sim: Optional[float] = None,
    ) -> MatchResult:
        forb = set(forbidden_ids or [])
        emb = l2_normalize(emb)

        if len(self._prototypes) == 0:
            return self._create_new(emb, similarity_hint=1.0) if create_new else MatchResult(-1, 1.0, False)

        ids = [pid for pid in self._prototypes.keys() if pid not in forb]
        if len(ids) == 0:
            return self._create_new(emb, similarity_hint=0.0) if create_new else MatchResult(-1, 0.0, False)

        protos = np.stack([self._prototypes[i] for i in ids], axis=0)  # (M, D)
        sims = protos @ emb  # (M,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = ids[best_idx]

        thr = self.sim_threshold if min_sim is None else float(min_sim)
        if best_sim >= thr:
            return MatchResult(best_id, best_sim, created_new=False)

        if create_new:
            return self._create_new(emb, similarity_hint=best_sim)

        return MatchResult(-1, best_sim, created_new=False)

    def similarity(self, person_id: int, emb: np.ndarray) -> float:
        proto = self._prototypes[person_id]
        emb = l2_normalize(emb)
        return float(proto @ emb)

    def should_update(self, similarity: float) -> bool:
        return float(similarity) >= self.update_threshold

    def update(self, person_id: int, emb: np.ndarray) -> None:
        proto = self._prototypes[person_id]
        emb = l2_normalize(emb)
        updated = self.ema * proto + (1.0 - self.ema) * emb
        self._prototypes[person_id] = l2_normalize(updated).astype(np.float32, copy=False)