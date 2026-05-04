from __future__ import annotations

from typing import Any

from pipeline.config import (
    DEFAULT_PIPELINE_CONFIG_TEXT,
    SUPPORTED_SOURCE_TYPES,
    DetectorConfig,
    ExtractorConfig,
    GalleryConfig,
    OutputConfig,
    PipelineConfig,
    ReidRunConfig,
    SourceConfig,
    StopConfig,
    TrackerConfig,
    dump_default_config,
    load_pipeline_config,
    merge_config_overrides,
    resolve_runtime_base_dir,
    write_default_config,
)

__all__ = [
    "DEFAULT_PIPELINE_CONFIG_TEXT",
    "SUPPORTED_SOURCE_TYPES",
    "DetectorConfig",
    "ExtractorConfig",
    "GalleryConfig",
    "OutputConfig",
    "PipelineConfig",
    "ReidRunConfig",
    "SourceConfig",
    "StopConfig",
    "TrackerConfig",
    "build_runtime",
    "dump_default_config",
    "load_pipeline_config",
    "merge_config_overrides",
    "resolve_runtime_base_dir",
    "run_file",
    "run_live",
    "write_default_config",
]


def build_runtime(*args: Any, **kwargs: Any):
    from pipeline.reid.factory import build_runtime as _build_runtime

    return _build_runtime(*args, **kwargs)


def run_file(*args: Any, **kwargs: Any):
    from pipeline.run_reid_file import run_file as _run_file

    return _run_file(*args, **kwargs)


def run_live(*args: Any, **kwargs: Any):
    from pipeline.run_reid_live import run_live as _run_live

    return _run_live(*args, **kwargs)
