from dataclasses import dataclass
from typing import Dict, Optional, Set
import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


@dataclass
class MatchResult:
    person_id: int
    similarity: float


class ReIDGallery:
    def __init__(self, sim_threshold: float = 0.55, ema: float = 0.8):
        self.sim_threshold = sim_threshold
        self.ema = ema
        self._next_id = 1
        self._prototypes: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._prototypes)

    def _create_new(self, emb: np.ndarray, similarity_hint: float = 0.0) -> MatchResult:
        pid = self._next_id
        self._next_id += 1
        self._prototypes[pid] = emb.copy()
        return MatchResult(pid, float(similarity_hint))

    def match(self, emb: np.ndarray, forbidden_ids: Optional[Set[int]] = None) -> MatchResult:
        if forbidden_ids is None:
            forbidden_ids = set()

        if len(self._prototypes) == 0:
            return self._create_new(emb, similarity_hint=1.0)

        ids = [pid for pid in self._prototypes.keys() if pid not in forbidden_ids]
        if len(ids) == 0:
            return self._create_new(emb, similarity_hint=0.0)

        protos = np.stack([self._prototypes[i] for i in ids], axis=0)  # (M, D)
        sims = protos @ emb  # (M,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = ids[best_idx]

        if best_sim >= self.sim_threshold:
            return MatchResult(best_id, best_sim)

        return self._create_new(emb, similarity_hint=best_sim)

    def update(self, person_id: int, emb: np.ndarray) -> None:
        proto = self._prototypes[person_id]
        updated = self.ema * proto + (1.0 - self.ema) * emb
        updated /= (np.linalg.norm(updated) + 1e-12)
        self._prototypes[person_id] = updated.astype(np.float32, copy=False)
