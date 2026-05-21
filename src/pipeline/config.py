from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


SUPPORTED_SOURCE_TYPES = ("file", "usb", "camera", "rtsp", "http", "https", "mjpeg")

DEFAULT_PIPELINE_CONFIG_TEXT = """# Full pipeline configuration template.
# You can keep only the sections you want to override in your own TOML file.

[reid]
outputs_root = "outputs"
run_prefix = "reid"
output_video_name = "demo_reid.mp4"
log_every = 30
metrics_file_name = "metrics.json"

[reid.source]
# Available values: file, usb, camera, rtsp, http, https, mjpeg
type = "file"
# Leave empty to pass the source from the API/CLI.
uri = ""
# Only for usb/camera sources.
device_index = ""
reconnect = true
reconnect_delay_s = 3.0
reconnect_max_attempts = 5

[reid.output]
save_video = false
save_assignments = true

[reid.stop]
# Leave empty to disable the stop condition.
max_frames = ""
max_duration_s = ""

[reid.detector]
# Any Ultralytics model name or local weights can be used.
model_name = "yolo26n.pt"
weights_path = ""
conf = 0.45
classes = [0]

[reid.extractor]
# Any Torchreid model name can be used.
model_name = "osnet_x1_0"
weights_path = ""
# Examples: "", "cpu", "cuda", "cuda:0"
device = ""

[reid.gallery]
sim_threshold = 0.75
ema = 0.65
update_threshold = 0.80
max_ids = 1000
state_path = "outputs/reid_gallery.json"
load_on_start = false
save_on_exit = true
autosave_interval_s = 60.0

[reid.tracker]
iou_threshold = 0.5
max_missed = 3
confirm_hits = 3
"""

DEFAULT_PIPELINE_CONFIG_RAW: Dict[str, Any] = {
    "reid": {
        "input_video": "",
        "outputs_root": "outputs",
        "run_prefix": "reid",
        "output_video_name": "demo_reid.mp4",
        "log_every": 30,
        "metrics_file_name": "metrics.json",
        "source": {
            "type": "file",
            "uri": "",
            "device_index": "",
            "reconnect": True,
            "reconnect_delay_s": 3.0,
            "reconnect_max_attempts": 5,
        },
        "output": {
            "save_video": False,
            "save_assignments": True,
        },
        "stop": {
            "max_frames": "",
            "max_duration_s": "",
        },
        "detector": {
            "model_name": "yolo26n.pt",
            "weights_path": "",
            "conf": 0.45,
            "classes": [0],
        },
        "extractor": {
            "model_name": "osnet_x1_0",
            "device": "",
            "weights_path": "",
        },
        "gallery": {
            "sim_threshold": 0.75,
            "ema": 0.65,
            "update_threshold": 0.80,
            "max_ids": 1000,
            "state_path": "outputs/reid_gallery.json",
            "load_on_start": False,
            "save_on_exit": True,
            "autosave_interval_s": 60.0,
        },
        "tracker": {
            "iou_threshold": 0.5,
            "max_missed": 3,
            "confirm_hits": 3,
        },
    }
}


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
    save_assignments: bool


@dataclass(frozen=True)
class StopConfig:
    max_frames: Optional[int]
    max_duration_s: Optional[float]


@dataclass(frozen=True)
class ReidRunConfig:
    input_video: Optional[str]
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


def dump_default_config() -> str:
    return DEFAULT_PIPELINE_CONFIG_TEXT


def write_default_config(path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Config file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_default_config(), encoding="utf-8")
    return output_path.resolve()


def merge_config_overrides(*overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in overrides:
        if not item:
            continue
        _deep_merge_dicts(merged, item)
    return merged


def resolve_runtime_base_dir(
    config_path: str | Path | None = None,
    *,
    work_dir: str | Path | None = None,
) -> Path:
    normalized = _normalize_config_path(config_path)
    if normalized is not None:
        return normalized.parent.resolve()

    if config_path is not None:
        candidate = Path(config_path).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    if work_dir is not None:
        return Path(work_dir).expanduser().resolve()
    return Path.cwd().resolve()


def load_pipeline_config(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    raw = deepcopy(DEFAULT_PIPELINE_CONFIG_RAW)

    normalized = _normalize_config_path(config_path)
    if normalized is not None:
        _deep_merge_dicts(raw, _load_toml_file(normalized))

    if overrides:
        _deep_merge_dicts(raw, overrides)

    cfg = _pipeline_config_from_raw(raw)
    validate_pipeline_config(cfg)
    return cfg


def validate_pipeline_config(cfg: PipelineConfig) -> None:
    reid = cfg.reid

    if reid.source.type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported reid.source.type={reid.source.type!r}. "
            f"Available values: {', '.join(SUPPORTED_SOURCE_TYPES)}"
        )

    _validate_non_empty("reid.outputs_root", reid.outputs_root)
    _validate_non_empty("reid.run_prefix", reid.run_prefix)
    _validate_non_empty("reid.output_video_name", reid.output_video_name)
    _validate_non_empty("reid.metrics_file_name", reid.metrics_file_name)
    _validate_positive_int("reid.log_every", reid.log_every)

    _validate_float_range("reid.detector.conf", reid.detector.conf, min_value=0.0, max_value=1.0)
    _validate_non_empty("reid.detector.model_name", reid.detector.model_name)

    _validate_non_empty("reid.extractor.model_name", reid.extractor.model_name)

    _validate_float_range("reid.gallery.sim_threshold", reid.gallery.sim_threshold, min_value=0.0, max_value=1.0)
    _validate_float_range("reid.gallery.ema", reid.gallery.ema, min_value=0.0, max_value=1.0)
    _validate_float_range(
        "reid.gallery.update_threshold",
        reid.gallery.update_threshold,
        min_value=0.0,
        max_value=1.0,
    )

    if reid.gallery.max_ids is not None and reid.gallery.max_ids <= 0:
        raise ValueError("reid.gallery.max_ids must be a positive integer or empty")
    if reid.gallery.autosave_interval_s is not None and reid.gallery.autosave_interval_s <= 0:
        raise ValueError("reid.gallery.autosave_interval_s must be positive or empty")

    _validate_float_range("reid.tracker.iou_threshold", reid.tracker.iou_threshold, min_value=0.0, max_value=1.0)
    if reid.tracker.max_missed < 0:
        raise ValueError("reid.tracker.max_missed must be non-negative")
    _validate_positive_int("reid.tracker.confirm_hits", reid.tracker.confirm_hits)

    if reid.source.device_index is not None and reid.source.device_index < 0:
        raise ValueError("reid.source.device_index must be non-negative or empty")
    if reid.source.reconnect_delay_s < 0:
        raise ValueError("reid.source.reconnect_delay_s must be non-negative")
    if reid.source.reconnect_max_attempts < 0:
        raise ValueError("reid.source.reconnect_max_attempts must be non-negative")

    if reid.stop.max_frames is not None and reid.stop.max_frames <= 0:
        raise ValueError("reid.stop.max_frames must be positive or empty")
    if reid.stop.max_duration_s is not None and reid.stop.max_duration_s <= 0:
        raise ValueError("reid.stop.max_duration_s must be positive or empty")


def _pipeline_config_from_raw(raw: Mapping[str, Any]) -> PipelineConfig:
    reid_raw = raw.get("reid", {})
    reid = ReidRunConfig(
        input_video=_opt_str(reid_raw.get("input_video")),
        outputs_root=str(reid_raw.get("outputs_root", "outputs")),
        run_prefix=str(reid_raw.get("run_prefix", "reid")),
        output_video_name=str(reid_raw.get("output_video_name", "demo_reid.mp4")),
        log_every=int(reid_raw.get("log_every", 30)),
        metrics_file_name=str(reid_raw.get("metrics_file_name", "metrics.json")),
        source=_source_cfg(reid_raw),
        output=OutputConfig(
            save_video=bool(reid_raw.get("output", {}).get("save_video", False)),
            save_assignments=bool(reid_raw.get("output", {}).get("save_assignments", True)),
        ),
        stop=StopConfig(
            max_frames=_opt_int(reid_raw.get("stop", {}).get("max_frames")),
            max_duration_s=_opt_float(reid_raw.get("stop", {}).get("max_duration_s")),
        ),
        detector=_detector_cfg(reid_raw.get("detector", {})),
        extractor=ExtractorConfig(
            model_name=str(reid_raw.get("extractor", {}).get("model_name", "osnet_x1_0")),
            device=_opt_str(reid_raw.get("extractor", {}).get("device")),
            weights_path=_opt_str(reid_raw.get("extractor", {}).get("weights_path")),
        ),
        gallery=GalleryConfig(
            sim_threshold=float(reid_raw.get("gallery", {}).get("sim_threshold", 0.75)),
            ema=float(reid_raw.get("gallery", {}).get("ema", 0.65)),
            update_threshold=float(reid_raw.get("gallery", {}).get("update_threshold", 0.80)),
            max_ids=_opt_int(reid_raw.get("gallery", {}).get("max_ids", 1000)),
            state_path=_opt_str(reid_raw.get("gallery", {}).get("state_path", "outputs/reid_gallery.json")),
            load_on_start=bool(reid_raw.get("gallery", {}).get("load_on_start", False)),
            save_on_exit=bool(reid_raw.get("gallery", {}).get("save_on_exit", True)),
            autosave_interval_s=_opt_float(reid_raw.get("gallery", {}).get("autosave_interval_s", 60.0)),
        ),
        tracker=TrackerConfig(
            iou_threshold=float(reid_raw.get("tracker", {}).get("iou_threshold", 0.5)),
            max_missed=int(reid_raw.get("tracker", {}).get("max_missed", 3)),
            confirm_hits=int(reid_raw.get("tracker", {}).get("confirm_hits", 3)),
        ),
    )
    return PipelineConfig(reid=reid)


def _detector_cfg(raw: Mapping[str, Any]) -> DetectorConfig:
    return DetectorConfig(
        model_name=str(raw.get("model_name", "yolo26n.pt")),
        weights_path=_opt_str(raw.get("weights_path")),
        conf=float(raw.get("conf", 0.45)),
        classes=[int(v) for v in raw.get("classes", [0])],
    )


def _source_cfg(reid_raw: Mapping[str, Any]) -> SourceConfig:
    raw = reid_raw.get("source", {})
    return SourceConfig(
        type=str(raw.get("type", "file")).strip().lower(),
        uri=_opt_str(raw.get("uri", reid_raw.get("input_video"))),
        device_index=_opt_int(raw.get("device_index")),
        reconnect=bool(raw.get("reconnect", True)),
        reconnect_delay_s=float(raw.get("reconnect_delay_s", 3.0)),
        reconnect_max_attempts=int(raw.get("reconnect_max_attempts", 5)),
    )


def _normalize_config_path(config_path: str | Path | None) -> Optional[Path]:
    if config_path is None:
        return None

    candidate = Path(config_path).expanduser()
    if candidate.exists() and candidate.is_dir():
        candidate = candidate / "configs" / "pipeline.toml"
    return candidate.resolve()


def _load_toml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if tomllib is None:
        raise RuntimeError("tomllib is unavailable")
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def _deep_merge_dicts(base: Dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge_dicts(base[key], value)
            continue
        if isinstance(value, Mapping):
            base[key] = deepcopy(dict(value))
            continue
        if isinstance(value, list):
            base[key] = list(value)
            continue
        base[key] = value


def _validate_float_range(name: str, value: float, *, min_value: float, max_value: float) -> None:
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be in range [{min_value}, {max_value}]")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_empty(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must not be empty")


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
