from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str
    weights_path: Optional[str]
    conf: float
    classes: List[int]


@dataclass(frozen=True)
class ExtractorConfig:
    model_name: str
    device: Optional[str]


@dataclass(frozen=True)
class GalleryConfig:
    sim_threshold: float
    ema: float
    update_threshold: float


@dataclass(frozen=True)
class TrackerConfig:
    iou_threshold: float
    max_missed: int
    confirm_hits: int
    new_identity_candidate_id: int


@dataclass(frozen=True)
class ReidRunConfig:
    input_video: str
    outputs_root: str
    run_prefix: str
    output_video_name: str
    log_every: int
    metrics_file_name: str
    detector: DetectorConfig
    extractor: ExtractorConfig
    gallery: GalleryConfig
    tracker: TrackerConfig


@dataclass(frozen=True)
class PipelineConfig:
    reid: ReidRunConfig


def load_pipeline_config(project_root: Path, config_relpath: str = "configs/pipeline.toml") -> PipelineConfig:
    project_root = Path(project_root)
    config_path = project_root / config_relpath

    raw: Dict[str, Any] = {}
    if config_path.exists():
        if tomllib is None:
            raise RuntimeError("tomllib is unavailable; use Python 3.11+ to read .toml config")
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    reid_raw = raw.get("reid", {})

    reid = ReidRunConfig(
        input_video=reid_raw.get("input_video", "assets/videos/test.mp4"),
        outputs_root=reid_raw.get("outputs_root", "outputs"),
        run_prefix=reid_raw.get("run_prefix", "runs"),
        output_video_name=reid_raw.get("output_video_name", "demo_reid.avi"),
        log_every=int(reid_raw.get("log_every", 30)),
        metrics_file_name=reid_raw.get("metrics_file_name", "metrics.json"),
        detector=_detector_cfg(reid_raw.get("detector", {})),
        extractor=ExtractorConfig(
            model_name=reid_raw.get("extractor", {}).get("model_name", "osnet_x0_25"),
            device=_opt_str(reid_raw.get("extractor", {}).get("device")),
        ),
        gallery=GalleryConfig(
            sim_threshold=float(reid_raw.get("gallery", {}).get("sim_threshold", 0.55)),
            ema=float(reid_raw.get("gallery", {}).get("ema", 0.8)),
            update_threshold=float(reid_raw.get("gallery", {}).get("update_threshold", 0.60)),
        ),
        tracker=TrackerConfig(
            iou_threshold=float(reid_raw.get("tracker", {}).get("iou_threshold", 0.3)),
            max_missed=int(reid_raw.get("tracker", {}).get("max_missed", 15)),
            confirm_hits=int(reid_raw.get("tracker", {}).get("confirm_hits", 3)),
            new_identity_candidate_id=int(reid_raw.get("tracker", {}).get("new_identity_candidate_id", 0)),
        ),
    )

    return PipelineConfig(reid=reid)


def _detector_cfg(raw: Dict[str, Any]) -> DetectorConfig:
    return DetectorConfig(
        model_name=str(raw.get("model_name", "yolo26n.pt")),
        weights_path=_opt_str(raw.get("weights_path", "models/detectors/yolo26n.pt")),
        conf=float(raw.get("conf", 0.25)),
        classes=[int(v) for v in raw.get("classes", [0])],
    )


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
