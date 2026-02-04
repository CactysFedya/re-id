from dataclasses import dataclass
from typing import Dict
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

    def match(self, emb: np.ndarray) -> MatchResult:
        if len(self._prototypes) == 0:
            pid = self._next_id
            self._next_id += 1
            self._prototypes[pid] = emb.copy()
            return MatchResult(pid, 1.0)

        ids = list(self._prototypes.keys())
        protos = np.stack([self._prototypes[i] for i in ids])
        sims = protos @ emb

        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        best_id = ids[idx]

        if best_sim >= self.sim_threshold:
            return MatchResult(best_id, best_sim)

        pid = self._next_id
        self._next_id += 1
        self._prototypes[pid] = emb.copy()
        return MatchResult(pid, best_sim)

    def update(self, person_id: int, emb: np.ndarray) -> None:
        proto = self._prototypes[person_id]
        updated = self.ema * proto + (1.0 - self.ema) * emb
        updated /= (np.linalg.norm(updated) + 1e-12)
        self._prototypes[person_id] = updated.astype(np.float32, copy=False)
