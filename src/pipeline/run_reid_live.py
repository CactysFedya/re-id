from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

from pipeline.config import load_pipeline_config, merge_config_overrides, resolve_runtime_base_dir
from pipeline.reid.factory import build_runtime
from pipeline.reid.persistence import (
    load_gallery_state,
    maybe_autosave_gallery_state,
    resolve_gallery_state_path,
    save_config_snapshot,
    save_gallery_state,
)
from pipeline.utils.logging import setup_logging
from pipeline.utils.paths import make_run_dir, resolve_path
from pipeline.utils.sources import is_live_source, open_capture, reconnect_capture, resolve_source_uri, source_label
from pipeline.utils.video import LatestFrameCapture, TimelineVideoRecorder, get_video_props


def run_live(
    source_uri: str | None = None,
    *,
    source_type: str | None = None,
    device_index: int | None = None,
    config_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> Path:
    base_dir = resolve_runtime_base_dir(config_path, work_dir=work_dir)
    overrides = merge_config_overrides(
        config_overrides,
        _build_live_source_override(source_uri, source_type, device_index),
    )
    cfg = load_pipeline_config(config_path, overrides=overrides).reid

    resolved_source_type = cfg.source.type
    if not is_live_source(resolved_source_type):
        raise ValueError(
            "run_live requires a live source type. "
            "Use usb, camera, rtsp, http, https, or mjpeg, or call run_file for video files."
        )
    resolved_source_uri = resolve_source_uri(base_dir, resolved_source_type, cfg.source.uri)
    source_label_value = source_label(resolved_source_type, resolved_source_uri, cfg.source.device_index)

    outputs_root = resolve_path(base_dir, cfg.outputs_root)
    if outputs_root is None:
        raise ValueError("outputs_root is required")

    run_dir = make_run_dir(outputs_root, prefix=f"{cfg.run_prefix}_live")
    requested_output_video = run_dir / cfg.output_video_name
    metrics_path = run_dir / cfg.metrics_file_name
    assignments_path = run_dir / "assignments.csv"
    config_snapshot_path = run_dir / "config_snapshot.json"
    gallery_state_path = resolve_gallery_state_path(base_dir, cfg)

    logger = setup_logging(log_file=run_dir / "run.log", name="pipeline.reid.live")
    runtime = build_runtime(base_dir, cfg)
    try:
        runtime.set_gallery(load_gallery_state(base_dir, cfg))
    except Exception as exc:
        logger.warning(f"Failed to load gallery state from {gallery_state_path}: {exc}")

    logger.info(f"Source:                {source_label_value}")
    logger.info(f"Run dir:               {run_dir}")
    logger.info(f"Save video:            {cfg.output.save_video}")
    logger.info(f"Save assignments:      {cfg.output.save_assignments}")
    if cfg.output.save_video:
        logger.info(f"Requested output video: {requested_output_video}")
    logger.info(f"Metrics file:          {metrics_path}")
    if cfg.output.save_assignments:
        logger.info(f"Assignments file:      {assignments_path}")
    logger.info(f"Config file:           {config_snapshot_path}")
    logger.info(f"Gallery file:          {gallery_state_path}")
    logger.info(f"Gallery size at start: {len(runtime.gallery)}")

    save_config_snapshot(config_snapshot_path, cfg)

    cap = open_capture(cfg, resolved_source_type, resolved_source_uri)
    props = get_video_props(cap)
    capture = LatestFrameCapture(cap)
    capture.start()
    recorder = TimelineVideoRecorder(run_dir / "_frames_live") if cfg.output.save_video else None

    assignments_file = None
    assignments_writer = None
    if cfg.output.save_assignments:
        assignments_file = assignments_path.open("w", newline="", encoding="utf-8")
        assignments_writer = csv.DictWriter(
            assignments_file,
            fieldnames=["frame", "track_id", "global_id", "cls", "conf", "x", "y", "w", "h"],
        )
        assignments_writer.writeheader()

    log_every = cfg.log_every
    start = time.time()
    start_perf = time.perf_counter()
    stop_reason = "unknown"
    recorded_fps = 0.0
    saved_video = None

    try:
        while True:
            elapsed_before_frame = time.time() - start
            if cfg.stop.max_duration_s is not None and elapsed_before_frame >= cfg.stop.max_duration_s:
                stop_reason = "max_duration_reached"
                logger.info(f"Stopping: reached max_duration_s={cfg.stop.max_duration_s}")
                break

            read_started = time.perf_counter()
            ok, frame = capture.read(timeout_s=max(1.0, cfg.source.reconnect_delay_s))
            if not ok:
                runtime.perf.add("read", time.perf_counter() - read_started)
                runtime.mark_skipped_frames(capture.consume_overwritten_count())

                capture.stop()
                cap.release()
                if not cfg.source.reconnect:
                    stop_reason = "source_exhausted"
                    logger.warning("Source returned no frame, stopping")
                    break

                cap = reconnect_capture(cfg, resolved_source_type, resolved_source_uri, logger)
                if cap is None:
                    stop_reason = "reconnect_exhausted"
                    logger.error("Reconnect attempts exhausted, stopping")
                    break
                runtime.reconnect_count += 1
                capture = LatestFrameCapture(cap)
                capture.start()
                continue
            runtime.perf.add("read", time.perf_counter() - read_started)

            runtime.mark_skipped_frames(capture.consume_overwritten_count())

            stats = runtime.process_frame(frame)
            if assignments_writer is not None:
                for row in runtime.last_assignments:
                    assignments_writer.writerow(row)

            if recorder is not None:
                write_started = time.perf_counter()
                recorder.add_frame(frame, time.perf_counter() - start_perf)
                runtime.perf.add("write", time.perf_counter() - write_started)
            else:
                runtime.perf.add("write", 0.0)

            autosaved_gallery_path = maybe_autosave_gallery_state(base_dir, cfg, runtime, time.perf_counter())
            if autosaved_gallery_path is not None:
                logger.info(
                    f"Autosaved gallery to: {autosaved_gallery_path} | "
                    f"autosave_ms={runtime.perf.avg_ms('autosave', runtime.frame_idx):.2f}"
                )

            if cfg.stop.max_frames is not None and runtime.frame_idx >= cfg.stop.max_frames:
                stop_reason = "max_frames_reached"
                logger.info(f"Stopping: reached max_frames={cfg.stop.max_frames}")
                break

            if runtime.frame_idx % log_every == 0:
                elapsed = time.time() - start
                fps_proc = runtime.frame_idx / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"[frame {runtime.frame_idx}] "
                    f"dets={stats['detections']} | tracks_now={stats['tracks_now']} | "
                    f"tracks_updated={stats['tracks_updated']} | gallery={stats['gallery_size']} | "
                    f"reconnects={runtime.reconnect_count} | skipped={runtime.skipped_frame_count} | "
                    f"fps={fps_proc:.2f} | detect_ms={runtime.perf.avg_ms('detect', runtime.frame_idx):.2f} | "
                    f"reid_ms={runtime.perf.avg_ms('reid', runtime.frame_idx):.2f}"
                )
    except KeyboardInterrupt:
        stop_reason = "user_interrupt"
        logger.info("Interrupted by user")
    finally:
        capture.stop()
        cap.release()
        if assignments_file is not None:
            assignments_file.close()

    if stop_reason == "unknown":
        stop_reason = "completed"

    total_wall_time_s = time.time() - start
    if recorder is not None and runtime.frame_idx > 0:
        saved_video = recorder.finalize(requested_output_video, props, total_duration_s=total_wall_time_s)
        if saved_video is not None:
            recorded_fps = saved_video.fps
    metrics = runtime.build_metrics(total_wall_time_s)
    metrics["source_fps"] = round(float(props.fps), 4)
    if props.fps > 0:
        metrics["processing_to_source_fps_ratio"] = round(float(metrics["avg_fps"]) / float(props.fps), 4)
    if recorded_fps > 0:
        metrics["recorded_video_fps"] = round(recorded_fps, 4)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    saved_gallery_path = save_gallery_state(base_dir, cfg, runtime.gallery)

    logger.info(
        f"Finished {runtime.frame_idx} frames in {total_wall_time_s:.2f}s | "
        f"skipped={runtime.skipped_frame_count} | stop_reason={stop_reason}"
    )
    logger.info(
        f"Metrics | avg_fps={metrics['avg_fps']:.4f} | avg_total_ms={metrics['avg_total_ms']:.4f} | "
        f"avg_detect_ms={metrics.get('avg_detect_ms', 0.0):.4f} | avg_reid_ms={metrics.get('avg_reid_ms', 0.0):.4f} | "
        f"avg_write_ms={metrics.get('avg_write_ms', 0.0):.4f} | avg_autosave_ms={metrics.get('avg_autosave_ms', 0.0):.4f} | "
        f"total_autosaves={metrics['total_autosaves']} | total_gallery_evictions={metrics['total_gallery_evictions']} | "
        f"total_global_ids_created={metrics['total_global_ids_created']} | reappearance_count={metrics['reappearance_count']} | "
        f"cross_session_reappearance_count={metrics['cross_session_reappearance_count']} | "
        f"reconnect_count={metrics['reconnect_count']} | stop_reason={stop_reason}"
    )
    if saved_video is not None:
        if saved_video.path != requested_output_video:
            logger.info(
                "Adjusted output container for compatibility: "
                f"{requested_output_video.name} -> {saved_video.path.name}"
            )
        logger.info(f"Recorded video fps:   {recorded_fps:.2f} (fixed container fps, real timing preserved by frame holds)")
        logger.info(f"Recorded video codec: {saved_video.codec}")
        logger.info(f"Saved output to:      {saved_video.path}")
    logger.info(f"Saved metrics to:     {metrics_path}")
    if cfg.output.save_assignments:
        logger.info(f"Saved assignments to: {assignments_path}")
    if saved_gallery_path is not None:
        logger.info(f"Saved gallery to:     {saved_gallery_path}")

    return run_dir


def main() -> None:
    run_live()


def _build_live_source_override(
    source_uri: str | None,
    source_type: str | None,
    device_index: int | None,
) -> dict[str, Any]:
    source_override: dict[str, Any] = {}
    if source_uri is not None:
        source_override["uri"] = source_uri
    if source_type is not None:
        source_override["type"] = source_type
    if device_index is not None:
        source_override["device_index"] = device_index
    if not source_override:
        return {}
    return {"reid": {"source": source_override}}


if __name__ == "__main__":
    main()
