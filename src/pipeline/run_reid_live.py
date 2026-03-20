from pathlib import Path
import json
import time

from pipeline.config import load_pipeline_config
from pipeline.reid_runtime import build_runtime, save_config_snapshot
from pipeline.utils.logging import setup_logging
from pipeline.utils.paths import find_project_root, make_run_dir
from pipeline.utils.sources import open_capture, reconnect_capture, resolve_source_uri, source_label
from pipeline.utils.video import LatestFrameCapture, TimelineVideoRecorder, get_video_props


def run_live() -> None:
    project_root = find_project_root(Path(__file__))
    cfg = load_pipeline_config(project_root).reid

    source_type = cfg.source.type
    source_uri = resolve_source_uri(project_root, source_type, cfg.source.uri)
    source_label_value = source_label(source_type, source_uri, cfg.source.device_index)

    outputs_root = project_root / cfg.outputs_root
    run_dir = make_run_dir(outputs_root, prefix=f"{cfg.run_prefix}_live")
    output_video = run_dir / cfg.output_video_name
    metrics_path = run_dir / cfg.metrics_file_name
    config_snapshot_path = run_dir / "config_snapshot.json"

    logger = setup_logging(log_file=run_dir / "run.log", name="pipeline.reid.live")
    runtime = build_runtime(project_root, cfg)

    logger.info(f"Source:       {source_label_value}")
    logger.info(f"Run dir:      {run_dir}")
    logger.info(f"Save video:   {cfg.output.save_video}")
    logger.info(f"Output video: {output_video}")
    logger.info(f"Metrics file: {metrics_path}")
    logger.info(f"Config file:  {config_snapshot_path}")

    save_config_snapshot(config_snapshot_path, cfg)

    cap = open_capture(cfg, source_type, source_uri)
    props = get_video_props(cap)
    capture = LatestFrameCapture(cap)
    capture.start()
    recorder = TimelineVideoRecorder(run_dir / "_frames_live") if cfg.output.save_video else None

    log_every = cfg.log_every
    start = time.time()
    start_perf = time.perf_counter()
    stop_reason = "unknown"
    recorded_fps = 0.0

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

                cap = reconnect_capture(cfg, source_type, source_uri, logger)
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

            if recorder is not None:
                write_started = time.perf_counter()
                recorder.add_frame(frame, time.perf_counter() - start_perf)
                runtime.perf.add("write", time.perf_counter() - write_started)
            else:
                runtime.perf.add("write", 0.0)

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

    if stop_reason == "unknown":
        stop_reason = "completed"

    total_wall_time_s = time.time() - start
    if recorder is not None and runtime.frame_idx > 0:
        recorded_fps = recorder.finalize(output_video, props, total_duration_s=total_wall_time_s)
    metrics = runtime.build_metrics(total_wall_time_s)
    if recorded_fps > 0:
        metrics["recorded_video_fps"] = round(recorded_fps, 4)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        f"Finished {runtime.frame_idx} frames in {total_wall_time_s:.2f}s | "
        f"skipped={runtime.skipped_frame_count} | stop_reason={stop_reason}"
    )
    logger.info(
        f"Metrics | avg_fps={metrics['avg_fps']:.4f} | avg_total_ms={metrics['avg_total_ms']:.4f} | "
        f"avg_detect_ms={metrics.get('avg_detect_ms', 0.0):.4f} | avg_reid_ms={metrics.get('avg_reid_ms', 0.0):.4f} | "
        f"avg_write_ms={metrics.get('avg_write_ms', 0.0):.4f} | total_global_ids_created={metrics['total_global_ids_created']} | "
        f"reappearance_count={metrics['reappearance_count']} | reconnect_count={metrics['reconnect_count']} | "
        f"stop_reason={stop_reason}"
    )
    if recorder is not None:
        logger.info(f"Recorded video fps: {recorded_fps:.2f} (fixed container fps, real timing preserved by frame holds)")
        logger.info(f"Saved output to: {output_video}")
    logger.info(f"Saved metrics to: {metrics_path}")


def main() -> None:
    run_live()


if __name__ == "__main__":
    main()
