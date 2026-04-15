from __future__ import annotations

from pathlib import Path

from pipeline.config import ReidRunConfig
from pipeline.detection.yolo import YoloDetector
from pipeline.reid.extractor import ReIDExtractor
from pipeline.reid.gallery import ReIDGallery
from pipeline.reid_runtime import ReidRuntime
from pipeline.tracking.iou import IoUTracker


def build_runtime(project_root: Path, cfg: ReidRunConfig) -> ReidRuntime:
    local_weights = project_root / cfg.detector.weights_path if cfg.detector.weights_path else None
    extractor_weights = project_root / cfg.extractor.weights_path if cfg.extractor.weights_path else None
    if extractor_weights is not None and not extractor_weights.exists():
        raise FileNotFoundError(f"ReID weights file not found: {extractor_weights}")

    detector = YoloDetector(
        model_name=cfg.detector.model_name,
        weights_path=local_weights,
        conf=cfg.detector.conf,
        classes=cfg.detector.classes,
    )
    extractor = ReIDExtractor(
        device=cfg.extractor.device,
        model_name=cfg.extractor.model_name,
        model_weights_path=str(extractor_weights) if extractor_weights else None,
    )
    gallery = ReIDGallery(
        sim_threshold=cfg.gallery.sim_threshold,
        ema=cfg.gallery.ema,
        update_threshold=cfg.gallery.update_threshold,
        max_ids=cfg.gallery.max_ids,
    )
    tracker = IoUTracker(iou_threshold=cfg.tracker.iou_threshold, max_missed=cfg.tracker.max_missed)
    return ReidRuntime(
        detector=detector,
        extractor=extractor,
        gallery=gallery,
        tracker=tracker,
        confirm_hits=cfg.tracker.confirm_hits,
    )
