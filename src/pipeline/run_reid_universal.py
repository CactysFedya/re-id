from pathlib import Path
import json
import time

import cv2

from pipeline.config import load_pipeline_config
from pipeline.utils.logging import setup_logging
from pipeline.utils.paths import find_project_root, make_run_dir
from pipeline.utils.sources import (
    build_reid_config_snapshot,
    is_live_source,
    open_capture,
    reconnect_capture,
    resolve_source_uri,
    source_label,
)
from pipeline.utils.video import get_video_props, open_writer_avi_mjpg
from pipeline.detection.yolo import YoloDetector
from pipeline.reid.extractor import ReIDExtractor
from pipeline.reid.gallery import ReIDGallery, l2_normalize
from pipeline.tracking.iou import IoUTracker


def main() -> None:
    project_root = find_project_root(Path(__file__))
    cfg = load_pipeline_config(project_root).reid

    outputs_root = project_root / cfg.outputs_root
    run_dir = make_run_dir(outputs_root, prefix=f"{cfg.run_prefix}_universal")
    output_video = run_dir / cfg.output_video_name
    metrics_path = run_dir / cfg.metrics_file_name
    config_snapshot_path = run_dir / "config_snapshot.json"
    logger = setup_logging(log_file=run_dir / "run.log", name="pipeline.universal")

    source_type = cfg.source.type
    source_uri = resolve_source_uri(project_root, source_type, cfg.source.uri)
    source_label_value = source_label(source_type, source_uri, cfg.source.device_index)
    is_live = is_live_source(source_type)

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
    )
    tracker = IoUTracker(iou_threshold=cfg.tracker.iou_threshold, max_missed=cfg.tracker.max_missed)
    confirm_hits = cfg.tracker.confirm_hits
    new_identity_candidate_id = cfg.tracker.new_identity_candidate_id

    logger.info(f"Source:       {source_label_value}")
    logger.info(f"Run dir:      {run_dir}")
    logger.info(f"Save video:   {cfg.output.save_video}")
    logger.info(f"Output video: {output_video}")
    logger.info(f"Metrics file: {metrics_path}")
    logger.info(f"Config file:  {config_snapshot_path}")

    config_snapshot = build_reid_config_snapshot(cfg)
    config_snapshot_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    cap = open_capture(cfg, source_type, source_uri)
    props = get_video_props(cap)
    out = open_writer_avi_mjpg(output_video, props) if cfg.output.save_video else None

    log_every = cfg.log_every
    start = time.time()
    frame_idx = 0
    total_frame_time_s = 0.0
    total_feature_time_s = 0.0
    reconnect_count = 0
    reappearance_count = 0
    person_last_track = {}
    stop_reason = "unknown"
    draw_color = (0, 255, 0)
    draw_box_thickness = 2
    draw_font_scale = 0.6
    draw_text_thickness = 2

    try:
        while True:
            elapsed_before_frame = time.time() - start
            if cfg.stop.max_duration_s is not None and elapsed_before_frame >= cfg.stop.max_duration_s:
                stop_reason = "max_duration_reached"
                logger.info(f"Stopping: reached max_duration_s={cfg.stop.max_duration_s}")
                break

            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if not is_live or not cfg.source.reconnect:
                    stop_reason = "source_exhausted"
                    logger.warning("Source returned no frame, stopping")
                    break

                cap.release()
                cap = reconnect_capture(cfg, source_type, source_uri, logger)
                if cap is None:
                    stop_reason = "reconnect_exhausted"
                    logger.error("Reconnect attempts exhausted, stopping")
                    break
                reconnect_count += 1
                continue

            dets = detector.predict(frame)
            track_dets = tracker.update(dets)

            crops = []
            items = []
            for td in track_dets:
                x1, y1, x2, y2 = td.bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                items.append((td.track_id, x1, y1, x2, y2, td.conf))

            used_person_ids = set()
            used_candidate_ids = set()

            feature_time_s = 0.0

            if crops:
                feature_started = time.perf_counter()
                feats = l2_normalize(extractor(crops))
                feature_time_s += time.perf_counter() - feature_started
                for i, (track_id, x1, y1, x2, y2, conf) in enumerate(items):
                    emb = feats[i]

                    pid = tracker.get_person_id(track_id)

                    if pid is not None:
                        sim = gallery.similarity(pid, emb)
                        if gallery.should_update(sim):
                            gallery.update(pid, emb)
                        used_person_ids.add(pid)
                    else:
                        forbidden = used_person_ids | used_candidate_ids
                        match = gallery.match(emb, forbidden_ids=forbidden, create_new=False)

                        cand_id = match.person_id
                        cand_sim = match.similarity

                        if cand_id == -1:
                            cand_id = new_identity_candidate_id
                            cand_sim = float("-inf")

                        if cand_id != new_identity_candidate_id:
                            confirmed = tracker.propose_person_id(track_id, cand_id, confirm_hits)
                            if confirmed is not None:
                                pid = confirmed
                                used_person_ids.add(pid)
                                if gallery.should_update(cand_sim):
                                    gallery.update(pid, emb)
                                prev_track_id = person_last_track.get(pid)
                                if prev_track_id is not None and prev_track_id != track_id:
                                    reappearance_count += 1
                                person_last_track[pid] = track_id
                            else:
                                used_candidate_ids.add(cand_id)
                        else:
                            confirmed = tracker.propose_person_id(track_id, new_identity_candidate_id, confirm_hits)
                            if confirmed is not None:
                                pid = gallery.add(emb)
                                tracker.set_person_id(track_id, pid)
                                used_person_ids.add(pid)
                                person_last_track[pid] = track_id
                            else:
                                used_candidate_ids.add(new_identity_candidate_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, draw_box_thickness)
                    cv2.putText(
                        frame,
                        f"tid {track_id} | id {pid} | det {conf:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        draw_font_scale,
                        draw_color,
                        draw_text_thickness,
                    )

            total_feature_time_s += feature_time_s
            if out is not None:
                out.write(frame)
            frame_idx += 1
            total_frame_time_s += time.perf_counter() - frame_started

            if cfg.stop.max_frames is not None and frame_idx >= cfg.stop.max_frames:
                stop_reason = "max_frames_reached"
                logger.info(f"Stopping: reached max_frames={cfg.stop.max_frames}")
                break

            if frame_idx % log_every == 0:
                elapsed = time.time() - start
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
                suffix = f"/{props.frame_count}" if props.frame_count > 0 else ""
                logger.info(
                    f"[frame {frame_idx}{suffix}] "
                    f"dets={len(dets)} | tracks_now={len(tracker.tracks())} | "
                    f"tracks_updated={len(track_dets)} | gallery={len(gallery)} | "
                    f"reconnects={reconnect_count} | fps={fps_proc:.2f}"
                )
    except KeyboardInterrupt:
        stop_reason = "user_interrupt"
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if out is not None:
            out.release()

    if stop_reason == "unknown":
        stop_reason = "completed"

    total_wall_time_s = time.time() - start
    avg_fps = frame_idx / total_wall_time_s if total_wall_time_s > 0 else 0.0
    avg_total_ms = (total_frame_time_s / frame_idx * 1000.0) if frame_idx > 0 else 0.0
    avg_feature_ms = (total_feature_time_s / frame_idx * 1000.0) if frame_idx > 0 else 0.0
    total_global_ids_created = gallery.total_ids_created()

    metrics = {
        "source_type": source_type,
        "source_label": source_label_value,
        "stop_reason": stop_reason,
        "frames_processed": frame_idx,
        "avg_fps": round(avg_fps, 4),
        "avg_total_ms": round(avg_total_ms, 4),
        "avg_feature_ms": round(avg_feature_ms, 4),
        "total_global_ids_created": int(total_global_ids_created),
        "reappearance_count": int(reappearance_count),
        "reconnect_count": int(reconnect_count),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Finished {frame_idx} frames in {total_wall_time_s:.2f}s | stop_reason={stop_reason}")
    logger.info(
        f"Metrics | avg_fps={metrics['avg_fps']:.4f} | avg_total_ms={metrics['avg_total_ms']:.4f} | "
        f"avg_feature_ms={metrics['avg_feature_ms']:.4f} | total_global_ids_created={metrics['total_global_ids_created']} | "
        f"reappearance_count={metrics['reappearance_count']} | reconnect_count={metrics['reconnect_count']} | "
        f"stop_reason={metrics['stop_reason']}"
    )
    if out is not None:
        logger.info(f"Saved output to: {output_video}")
    logger.info(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
