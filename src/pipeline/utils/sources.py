from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from pipeline.utils.video import open_video_source


LIVE_SOURCE_TYPES = {"usb", "camera", "rtsp", "http", "https", "mjpeg"}


def is_live_source(source_type: str) -> bool:
    return source_type in LIVE_SOURCE_TYPES


def resolve_source_uri(project_root: Path, source_type: str, source_uri: str | None) -> str | None:
    if source_type != "file" or source_uri is None:
        return source_uri

    candidate = Path(source_uri)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Put a video here: {candidate}")
    return str(candidate)


def source_label(source_type: str, source_uri: str | None, device_index: int | None) -> str:
    if source_type == "file":
        return source_uri or "<missing-file>"
    if source_type in {"usb", "camera"}:
        return f"camera:{0 if device_index is None else device_index}"
    return source_uri or f"<{source_type}>"


def open_capture(cfg: Any, source_type: str, source_uri: str | None) -> cv2.VideoCapture:
    return open_video_source(
        source_type=source_type,
        uri=source_uri,
        device_index=cfg.source.device_index,
    )


def reconnect_capture(cfg: Any, source_type: str, source_uri: str | None, logger: Any) -> cv2.VideoCapture | None:
    max_attempts = max(0, cfg.source.reconnect_max_attempts)
    for attempt in range(1, max_attempts + 1):
        logger.warning(
            f"Reconnect attempt {attempt}/{max_attempts} in {cfg.source.reconnect_delay_s:.1f}s"
        )
        time.sleep(cfg.source.reconnect_delay_s)
        try:
            return open_capture(cfg, source_type, source_uri)
        except Exception as exc:
            logger.warning(f"Reconnect failed: {exc}")
    return None


def build_reid_config_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "reid": {
            "input_video": cfg.input_video,
            "outputs_root": cfg.outputs_root,
            "run_prefix": cfg.run_prefix,
            "output_video_name": cfg.output_video_name,
            "log_every": cfg.log_every,
            "metrics_file_name": cfg.metrics_file_name,
            "source": {
                "type": cfg.source.type,
                "uri": cfg.source.uri,
                "device_index": cfg.source.device_index,
                "reconnect": cfg.source.reconnect,
                "reconnect_delay_s": cfg.source.reconnect_delay_s,
                "reconnect_max_attempts": cfg.source.reconnect_max_attempts,
            },
            "output": {
                "save_video": cfg.output.save_video,
            },
            "stop": {
                "max_frames": cfg.stop.max_frames,
                "max_duration_s": cfg.stop.max_duration_s,
            },
            "detector": {
                "model_name": cfg.detector.model_name,
                "weights_path": cfg.detector.weights_path,
                "conf": cfg.detector.conf,
                "classes": cfg.detector.classes,
            },
            "extractor": {
                "model_name": cfg.extractor.model_name,
                "device": cfg.extractor.device,
                "weights_path": cfg.extractor.weights_path,
            },
            "gallery": {
                "sim_threshold": cfg.gallery.sim_threshold,
                "ema": cfg.gallery.ema,
                "update_threshold": cfg.gallery.update_threshold,
            },
            "tracker": {
                "iou_threshold": cfg.tracker.iou_threshold,
                "max_missed": cfg.tracker.max_missed,
                "confirm_hits": cfg.tracker.confirm_hits,
                "new_identity_candidate_id": cfg.tracker.new_identity_candidate_id,
            },
        }
    }
