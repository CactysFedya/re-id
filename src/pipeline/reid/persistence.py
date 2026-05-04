from __future__ import annotations

import json
import time
from pathlib import Path

from pipeline.config import ReidRunConfig
from pipeline.reid.gallery import ReIDGallery
from pipeline.reid_runtime import ReidRuntime
from pipeline.utils.paths import resolve_path
from pipeline.utils.sources import build_reid_config_snapshot


def save_config_snapshot(path: Path, cfg: ReidRunConfig) -> None:
    snapshot = build_reid_config_snapshot(cfg)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_gallery_state_path(base_dir: Path, cfg: ReidRunConfig) -> Path | None:
    return resolve_path(base_dir, cfg.gallery.state_path)


def load_gallery_state(base_dir: Path, cfg: ReidRunConfig) -> ReIDGallery:
    gallery = ReIDGallery(
        sim_threshold=cfg.gallery.sim_threshold,
        ema=cfg.gallery.ema,
        update_threshold=cfg.gallery.update_threshold,
        max_ids=cfg.gallery.max_ids,
    )
    if not cfg.gallery.load_on_start:
        return gallery

    state_path = resolve_gallery_state_path(base_dir, cfg)
    if state_path is None or not state_path.exists():
        return gallery

    loaded = ReIDGallery.load(state_path)

    metadata = loaded.metadata()
    saved_model = metadata.get("extractor_model_name")
    saved_weights = metadata.get("extractor_weights_path")
    if saved_model not in {None, cfg.extractor.model_name}:
        raise ValueError(
            f"Gallery extractor model mismatch: saved={saved_model}, current={cfg.extractor.model_name}"
        )
    if saved_weights not in {None, cfg.extractor.weights_path}:
        raise ValueError(
            f"Gallery extractor weights mismatch: saved={saved_weights}, current={cfg.extractor.weights_path}"
        )

    loaded.sim_threshold = float(cfg.gallery.sim_threshold)
    loaded.update_threshold = float(cfg.gallery.update_threshold)
    loaded.ema = float(cfg.gallery.ema)
    loaded.max_ids = cfg.gallery.max_ids
    loaded.enforce_capacity()
    return loaded


def _persist_gallery_state(base_dir: Path, cfg: ReidRunConfig, gallery: ReIDGallery) -> Path | None:
    state_path = resolve_gallery_state_path(base_dir, cfg)
    if state_path is None:
        return None

    gallery.save(
        state_path,
        metadata={
            "extractor_model_name": cfg.extractor.model_name,
            "extractor_weights_path": cfg.extractor.weights_path,
        },
    )
    return state_path


def save_gallery_state(base_dir: Path, cfg: ReidRunConfig, gallery: ReIDGallery) -> Path | None:
    if not cfg.gallery.save_on_exit:
        return None

    return _persist_gallery_state(base_dir, cfg, gallery)


def maybe_autosave_gallery_state(
    base_dir: Path,
    cfg: ReidRunConfig,
    runtime: ReidRuntime,
    now_monotonic_s: float,
) -> Path | None:
    interval_s = cfg.gallery.autosave_interval_s
    if interval_s is None or interval_s <= 0:
        return None

    if resolve_gallery_state_path(base_dir, cfg) is None:
        return None

    last = runtime.last_gallery_autosave_monotonic_s
    if last is None:
        runtime.last_gallery_autosave_monotonic_s = float(now_monotonic_s)
        return None

    if float(now_monotonic_s) - last < interval_s:
        return None

    started = time.perf_counter()
    saved_path = _persist_gallery_state(base_dir, cfg, runtime.gallery)
    runtime.perf.add("autosave", time.perf_counter() - started, count=1)
    runtime.last_gallery_autosave_monotonic_s = float(now_monotonic_s)
    return saved_path
