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
    weights_path: Optional[str]


@dataclass(frozen=True)
class GalleryConfig:
    sim_threshold: float
    ema: float
    update_threshold: float
    max_ids: Optional[int]
    state_path: Optional[str]
    load_on_start: bool
    save_on_exit: bool
    autosave_interval_s: Optional[float]


@dataclass(frozen=True)
class TrackerConfig:
    iou_threshold: float
    max_missed: int
    confirm_hits: int


@dataclass(frozen=True)
class SourceConfig:
    type: str
    uri: Optional[str]
    device_index: Optional[int]
    reconnect: bool
    reconnect_delay_s: float
    reconnect_max_attempts: int


@dataclass(frozen=True)
class OutputConfig:
    save_video: bool


@dataclass(frozen=True)
class StopConfig:
    max_frames: Optional[int]
    max_duration_s: Optional[float]


@dataclass(frozen=True)
class ReidRunConfig:
    input_video: str
    outputs_root: str
    run_prefix: str
    output_video_name: str
    log_every: int
    metrics_file_name: str
    source: SourceConfig
    output: OutputConfig
    stop: StopConfig
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
        source=_source_cfg(reid_raw),
        output=OutputConfig(
            save_video=bool(reid_raw.get("output", {}).get("save_video", True)),
        ),
        stop=StopConfig(
            max_frames=_opt_int(reid_raw.get("stop", {}).get("max_frames")),
            max_duration_s=_opt_float(reid_raw.get("stop", {}).get("max_duration_s")),
        ),
        detector=_detector_cfg(reid_raw.get("detector", {})),
        extractor=ExtractorConfig(
            model_name=reid_raw.get("extractor", {}).get("model_name", "osnet_x0_25"),
            device=_opt_str(reid_raw.get("extractor", {}).get("device")),
            weights_path=_opt_str(reid_raw.get("extractor", {}).get("weights_path")),
        ),
        gallery=GalleryConfig(
            sim_threshold=float(reid_raw.get("gallery", {}).get("sim_threshold", 0.55)),
            ema=float(reid_raw.get("gallery", {}).get("ema", 0.8)),
            update_threshold=float(reid_raw.get("gallery", {}).get("update_threshold", 0.60)),
            max_ids=_opt_int(reid_raw.get("gallery", {}).get("max_ids", 1000)),
            state_path=_opt_str(reid_raw.get("gallery", {}).get("state_path", "outputs/reid_gallery.json")),
            load_on_start=bool(reid_raw.get("gallery", {}).get("load_on_start", True)),
            save_on_exit=bool(reid_raw.get("gallery", {}).get("save_on_exit", True)),
            autosave_interval_s=_opt_float(reid_raw.get("gallery", {}).get("autosave_interval_s", 60.0)),
        ),
        tracker=TrackerConfig(
            iou_threshold=float(reid_raw.get("tracker", {}).get("iou_threshold", 0.3)),
            max_missed=int(reid_raw.get("tracker", {}).get("max_missed", 15)),
            confirm_hits=int(reid_raw.get("tracker", {}).get("confirm_hits", 3)),
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


def _source_cfg(reid_raw: Dict[str, Any]) -> SourceConfig:
    raw = reid_raw.get("source", {})
    return SourceConfig(
        type=str(raw.get("type", "file")).strip().lower(),
        uri=_opt_str(raw.get("uri", reid_raw.get("input_video", "assets/videos/test.mp4"))),
        device_index=_opt_int(raw.get("device_index")),
        reconnect=bool(raw.get("reconnect", True)),
        reconnect_delay_s=float(raw.get("reconnect_delay_s", 3.0)),
        reconnect_max_attempts=int(raw.get("reconnect_max_attempts", 5)),
    )


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
