from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipeline.config import (
    SUPPORTED_SOURCE_TYPES,
    dump_default_config,
    load_pipeline_config,
    write_default_config,
)

LIVE_SOURCE_TYPES = {"usb", "camera", "rtsp", "http", "https", "mjpeg"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_run_parser()
    args = parser.parse_args(argv)

    if args.video and args.source_uri:
        parser.error("Use either --video or --source-uri, not both.")
    if args.video and args.source_type and args.source_type != "file":
        parser.error("--video can only be used with source type 'file'.")

    if args.video is not None:
        from pipeline.run_reid_file import run_file

        run_file(
            video_path=args.video,
            config_path=args.config,
            work_dir=args.work_dir,
        )
        return

    if args.source_uri is not None or args.source_type is not None or args.device_index is not None:
        source_type = args.source_type or _infer_source_type(args.source_uri, args.device_index)
        if source_type == "file":
            from pipeline.run_reid_file import run_file

            run_file(
                video_path=args.source_uri,
                config_path=args.config,
                work_dir=args.work_dir,
            )
            return

        from pipeline.run_reid_live import run_live

        run_live(
            source_uri=args.source_uri,
            source_type=source_type,
            device_index=args.device_index,
            config_path=args.config,
            work_dir=args.work_dir,
        )
        return

    cfg = load_pipeline_config(args.config).reid
    if cfg.source.type in LIVE_SOURCE_TYPES:
        from pipeline.run_reid_live import run_live

        run_live(config_path=args.config, work_dir=args.work_dir)
        return
    from pipeline.run_reid_file import run_file

    run_file(config_path=args.config, work_dir=args.work_dir)


def dump_config_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="reid-dump-config",
        description="Print the full default pipeline configuration or save it to a file.",
    )
    parser.add_argument("output", nargs="?", help="Optional path to write the template TOML file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    args = parser.parse_args(argv)

    if args.output:
        output_path = write_default_config(args.output, overwrite=args.force)
        print(output_path)
        return

    print(dump_default_config(), end="")


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reid-run",
        description="Run the configurable ReID pipeline from built-in defaults or a user TOML file.",
    )
    parser.add_argument(
        "--config",
        help="Path to a partial or full pipeline TOML config. Omit it to use built-in defaults.",
    )
    parser.add_argument(
        "--work-dir",
        help="Base directory used for relative paths when --config is not provided.",
    )
    parser.add_argument(
        "--video",
        help="Video path for file mode. Equivalent to --source-type file --source-uri <path>.",
    )
    parser.add_argument(
        "--source-uri",
        help="Source URI or path override. Examples: rtsp://..., http://..., assets/demo.mp4.",
    )
    parser.add_argument(
        "--source-type",
        choices=SUPPORTED_SOURCE_TYPES,
        help="Explicit source type override.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help="Camera index override for usb/camera sources.",
    )
    return parser


def _infer_source_type(source_uri: str | None, device_index: int | None) -> str:
    if device_index is not None and source_uri is None:
        return "camera"
    if source_uri is None:
        return "file"

    lowered = source_uri.lower()
    if lowered.startswith("rtsp://"):
        return "rtsp"
    if lowered.startswith("https://"):
        return "https"
    if lowered.startswith("http://"):
        return "http"
    return "file"
