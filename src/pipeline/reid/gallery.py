from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional
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
    identity_id: int
    similarity: float
    created_new: bool = False


@dataclass
class IdentityMetadata:
    created_at_s: float
    last_seen_at_s: float
    last_update_at_s: float
    seen_count: int = 1
    update_count: int = 0
    label: Optional[int] = None

    def to_state(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict[str, Any], *, fallback_label: Optional[int] = None) -> "IdentityMetadata":
        created_at_s = float(state.get("created_at_s", time.time()))
        last_seen_at_s = float(state.get("last_seen_at_s", created_at_s))
        last_update_at_s = float(state.get("last_update_at_s", created_at_s))
        seen_count = int(state.get("seen_count", 1))
        update_count = int(state.get("update_count", 0))
        label = state.get("label", fallback_label)
        return cls(
            created_at_s=created_at_s,
            last_seen_at_s=last_seen_at_s,
            last_update_at_s=last_update_at_s,
            seen_count=seen_count,
            update_count=update_count,
            label=None if label is None else int(label),
        )


class ReIDGallery:
    def __init__(
        self,
        sim_threshold: float = 0.55,
        ema: float = 0.8,
        update_threshold: Optional[float] = None,
        max_ids: Optional[int] = None,
    ):
        self.sim_threshold = float(sim_threshold)
        self.update_threshold = float(update_threshold) if update_threshold is not None else float(sim_threshold)
        self.ema = float(ema)
        self.max_ids = int(max_ids) if max_ids is not None else None

        self._next_id = 1
        self._prototypes: Dict[int, np.ndarray] = {}
        self._labels: Dict[int, int] = {}
        self._identity_meta: Dict[int, IdentityMetadata] = {}
        self._metadata: dict = {}
        self._eviction_count = 0

    def __len__(self) -> int:
        return len(self._prototypes)

    def total_ids_created(self) -> int:
        return self._next_id - 1

    def total_evictions(self) -> int:
        return int(self._eviction_count)

    def _now_s(self) -> float:
        return time.time()

    def _delete_identity(self, identity_id: int) -> None:
        self._prototypes.pop(identity_id, None)
        self._labels.pop(identity_id, None)
        self._identity_meta.pop(identity_id, None)

    def _eviction_key(self, identity_id: int) -> tuple[float, int, float, int]:
        meta = self._identity_meta[identity_id]
        return (meta.last_seen_at_s, meta.seen_count, meta.last_update_at_s, identity_id)

    def _evict_if_needed(self, protected_ids: Optional[Iterable[int]] = None) -> None:
        if self.max_ids is None or self.max_ids <= 0:
            return

        protected = set(protected_ids or [])
        while len(self._prototypes) >= self.max_ids:
            candidates = [identity_id for identity_id in self._prototypes if identity_id not in protected]
            if not candidates:
                return

            evict_id = min(candidates, key=self._eviction_key)
            self._delete_identity(evict_id)
            self._eviction_count += 1

    def enforce_capacity(self, protected_ids: Optional[Iterable[int]] = None) -> None:
        self._evict_if_needed(protected_ids=protected_ids)

    def _create_new(
        self,
        emb: np.ndarray,
        similarity_hint: float = 0.0,
        label: Optional[int] = None,
        protected_ids: Optional[Iterable[int]] = None,
    ) -> MatchResult:
        self._evict_if_needed(protected_ids=protected_ids)
        identity_id = self._next_id
        self._next_id += 1
        self._prototypes[identity_id] = l2_normalize(emb).copy()
        if label is not None:
            self._labels[identity_id] = int(label)
        now_s = self._now_s()
        self._identity_meta[identity_id] = IdentityMetadata(
            created_at_s=now_s,
            last_seen_at_s=now_s,
            last_update_at_s=now_s,
            seen_count=1,
            update_count=0,
            label=None if label is None else int(label),
        )
        return MatchResult(identity_id, float(similarity_hint), created_new=True)

    def add(self, emb: np.ndarray, label: Optional[int] = None, protected_ids: Optional[Iterable[int]] = None) -> int:
        return self._create_new(emb, label=label, protected_ids=protected_ids).identity_id

    def match(
        self,
        emb: np.ndarray,
        forbidden_ids: Optional[Iterable[int]] = None,
        *,
        create_new: bool = True,
        min_sim: Optional[float] = None,
        label: Optional[int] = None,
        protected_ids: Optional[Iterable[int]] = None,
    ) -> MatchResult:
        forb = set(forbidden_ids or [])
        emb = l2_normalize(emb)

        if len(self._prototypes) == 0:
            return (
                self._create_new(emb, similarity_hint=1.0, label=label, protected_ids=protected_ids)
                if create_new
                else MatchResult(-1, 1.0, False)
            )

        ids = [
            identity_id
            for identity_id in self._prototypes.keys()
            if identity_id not in forb and (label is None or self._labels.get(identity_id) == int(label))
        ]
        if len(ids) == 0:
            return (
                self._create_new(emb, similarity_hint=0.0, label=label, protected_ids=protected_ids)
                if create_new
                else MatchResult(-1, 0.0, False)
            )

        protos = np.stack([self._prototypes[i] for i in ids], axis=0)  # (M, D)
        sims = protos @ emb  # (M,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = ids[best_idx]

        thr = self.sim_threshold if min_sim is None else float(min_sim)
        if best_sim >= thr:
            return MatchResult(best_id, best_sim, created_new=False)

        if create_new:
            return self._create_new(emb, similarity_hint=best_sim, label=label, protected_ids=protected_ids)

        return MatchResult(-1, best_sim, created_new=False)

    def similarity(self, identity_id: int, emb: np.ndarray) -> float:
        proto = self._prototypes[identity_id]
        emb = l2_normalize(emb)
        return float(proto @ emb)

    def should_update(self, similarity: float) -> bool:
        return float(similarity) >= self.update_threshold

    def note_seen(self, identity_id: int) -> None:
        meta = self._identity_meta[identity_id]
        meta.last_seen_at_s = self._now_s()
        meta.seen_count += 1

    def update(self, identity_id: int, emb: np.ndarray) -> None:
        proto = self._prototypes[identity_id]
        emb = l2_normalize(emb)
        updated = self.ema * proto + (1.0 - self.ema) * emb
        self._prototypes[identity_id] = l2_normalize(updated).astype(np.float32, copy=False)
        meta = self._identity_meta[identity_id]
        meta.last_update_at_s = self._now_s()
        meta.update_count += 1

    def identity_metadata(self, identity_id: int) -> IdentityMetadata:
        meta = self._identity_meta[identity_id]
        return IdentityMetadata(
            created_at_s=meta.created_at_s,
            last_seen_at_s=meta.last_seen_at_s,
            last_update_at_s=meta.last_update_at_s,
            seen_count=meta.seen_count,
            update_count=meta.update_count,
            label=meta.label,
        )

    def identity_metadata_state(self, identity_id: int) -> dict[str, Any]:
        return self.identity_metadata(identity_id).to_state()

    def embedding_dim(self) -> int:
        if not self._prototypes:
            return 0
        return int(next(iter(self._prototypes.values())).shape[0])

    def to_state(self, *, metadata: Optional[dict] = None) -> dict:
        return {
            "version": 2,
            "sim_threshold": self.sim_threshold,
            "update_threshold": self.update_threshold,
            "ema": self.ema,
            "max_ids": self.max_ids,
            "next_id": self._next_id,
            "embedding_dim": self.embedding_dim(),
            "eviction_count": self._eviction_count,
            "identities": [
                {
                    "identity_id": int(identity_id),
                    "label": self._labels.get(identity_id),
                    "identity_metadata": self._identity_meta[identity_id].to_state(),
                    "prototype": proto.astype(np.float32, copy=False).tolist(),
                }
                for identity_id, proto in sorted(self._prototypes.items())
            ],
            "metadata": metadata or self._metadata,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ReIDGallery":
        version = int(state.get("version", 1))
        if version not in {1, 2}:
            raise ValueError(f"Unsupported gallery state version: {version}")

        gallery = cls(
            sim_threshold=float(state.get("sim_threshold", 0.55)),
            ema=float(state.get("ema", 0.8)),
            update_threshold=float(state.get("update_threshold", state.get("sim_threshold", 0.55))),
            max_ids=state.get("max_ids"),
        )

        identities = state.get("identities", [])
        expected_dim = int(state.get("embedding_dim", 0))
        for item in identities:
            identity_id = int(item["identity_id"])
            proto = l2_normalize(np.asarray(item["prototype"], dtype=np.float32))
            if proto.ndim != 1:
                raise ValueError(f"Expected 1D prototype for identity_id={identity_id}, got shape {proto.shape}")
            if expected_dim > 0 and int(proto.shape[0]) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for identity_id={identity_id}: "
                    f"expected {expected_dim}, got {proto.shape[0]}"
                )

            gallery._prototypes[identity_id] = proto
            label = item.get("label")
            if label is not None:
                gallery._labels[identity_id] = int(label)
            meta_state = item.get("identity_metadata", {})
            gallery._identity_meta[identity_id] = IdentityMetadata.from_state(meta_state, fallback_label=label)

        gallery._metadata = dict(state.get("metadata", {}))
        gallery._eviction_count = int(state.get("eviction_count", 0))
        next_id = int(state.get("next_id", 1))
        if gallery._prototypes:
            gallery._next_id = max(next_id, max(gallery._prototypes.keys()) + 1)
        else:
            gallery._next_id = max(1, next_id)
        return gallery

    def metadata(self) -> dict:
        return dict(self._metadata)

    def save(self, path: Path, *, metadata: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(metadata=metadata), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReIDGallery":
        path = Path(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_state(state)
