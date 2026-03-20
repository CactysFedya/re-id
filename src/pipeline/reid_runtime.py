from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any

import cv2

from pipeline.config import ReidRunConfig
from pipeline.detection.yolo import YoloDetector
from pipeline.reid.extractor import ReIDExtractor
from pipeline.reid.gallery import ReIDGallery, l2_normalize
from pipeline.tracking.iou import IoUTracker
from pipeline.utils.performance import StageTimer
from pipeline.utils.sources import build_reid_config_snapshot

@dataclass
class ReidRuntime:
    detector: YoloDetector
    extractor: ReIDExtractor
    gallery: ReIDGallery
    tracker: IoUTracker
    confirm_hits: int
    new_identity_candidate_id: int
    perf: StageTimer = field(default_factory=StageTimer)
    frame_idx: int = 0
    total_frame_time_s: float = 0.0
    reappearance_count: int = 0
    skipped_frame_count: int = 0
    reconnect_count: int = 0
    person_last_track: dict[int, int] = field(default_factory=dict)
    draw_color: tuple[int, int, int] = (0, 255, 0)
    draw_box_thickness: int = 2
    draw_font_scale: float = 0.6
    draw_text_thickness: int = 2

    def process_frame(self, frame: Any) -> dict[str, int]:
        frame_started = time.perf_counter()

        detect_started = time.perf_counter()
        dets = self.detector.predict(frame)
        self.perf.add("detect", time.perf_counter() - detect_started, count=len(dets))

        track_started = time.perf_counter()
        track_dets = self.tracker.update(dets)
        self.perf.add("track", time.perf_counter() - track_started, count=len(track_dets))

        crops = []
        items = []
        crop_started = time.perf_counter()
        for td in track_dets:
            x1, y1, x2, y2 = td.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            items.append((td.track_id, x1, y1, x2, y2, td.conf))
        self.perf.add("crop", time.perf_counter() - crop_started, count=len(crops))

        used_person_ids = set()
        used_candidate_ids = set()

        if crops:
            feature_started = time.perf_counter()
            feats = l2_normalize(self.extractor(crops))
            self.perf.add("reid", time.perf_counter() - feature_started, count=len(crops))

            draw_started = time.perf_counter()
            for i, (track_id, x1, y1, x2, y2, conf) in enumerate(items):
                emb = feats[i]
                pid = self.tracker.get_person_id(track_id)

                if pid is not None:
                    sim = self.gallery.similarity(pid, emb)
                    if self.gallery.should_update(sim):
                        self.gallery.update(pid, emb)
                    used_person_ids.add(pid)
                else:
                    forbidden = used_person_ids | used_candidate_ids
                    match = self.gallery.match(emb, forbidden_ids=forbidden, create_new=False)
                    cand_id = match.person_id
                    cand_sim = match.similarity

                    if cand_id == -1:
                        cand_id = self.new_identity_candidate_id
                        cand_sim = float("-inf")

                    if cand_id != self.new_identity_candidate_id:
                        confirmed = self.tracker.propose_person_id(track_id, cand_id, self.confirm_hits)
                        if confirmed is not None:
                            pid = confirmed
                            used_person_ids.add(pid)
                            if self.gallery.should_update(cand_sim):
                                self.gallery.update(pid, emb)
                            prev_track_id = self.person_last_track.get(pid)
                            if prev_track_id is not None and prev_track_id != track_id:
                                self.reappearance_count += 1
                            self.person_last_track[pid] = track_id
                        else:
                            used_candidate_ids.add(cand_id)
                    else:
                        confirmed = self.tracker.propose_person_id(
                            track_id,
                            self.new_identity_candidate_id,
                            self.confirm_hits,
                        )
                        if confirmed is not None:
                            pid = self.gallery.add(emb)
                            self.tracker.set_person_id(track_id, pid)
                            used_person_ids.add(pid)
                            self.person_last_track[pid] = track_id
                        else:
                            used_candidate_ids.add(self.new_identity_candidate_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), self.draw_color, self.draw_box_thickness)
                cv2.putText(
                    frame,
                    f"tid {track_id} | id {pid} | det {conf:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.draw_font_scale,
                    self.draw_color,
                    self.draw_text_thickness,
                )
            self.perf.add("draw", time.perf_counter() - draw_started, count=len(items))
        else:
            self.perf.add("reid", 0.0)
            self.perf.add("draw", 0.0)

        self.frame_idx += 1
        self.total_frame_time_s += time.perf_counter() - frame_started

        return {
            "detections": len(dets),
            "tracks_now": len(self.tracker.tracks()),
            "tracks_updated": len(track_dets),
            "gallery_size": len(self.gallery),
        }

    def mark_skipped_frames(self, count: int) -> None:
        self.skipped_frame_count += int(count)

    def build_metrics(self, total_wall_time_s: float) -> dict[str, float | int]:
        avg_fps = self.frame_idx / total_wall_time_s if total_wall_time_s > 0 else 0.0
        avg_total_ms = (self.total_frame_time_s / self.frame_idx * 1000.0) if self.frame_idx > 0 else 0.0
        return {
            "frames_processed": self.frame_idx,
            "frames_skipped": int(self.skipped_frame_count),
            "avg_fps": round(avg_fps, 4),
            "avg_total_ms": round(avg_total_ms, 4),
            **self.perf.summary(self.frame_idx, exclude={"loop"}),
            "total_detections": self.perf.total("detect"),
            "total_tracked_detections": self.perf.total("track"),
            "total_crops_encoded": self.perf.total("crop"),
            "total_global_ids_created": int(self.gallery.total_ids_created()),
            "reappearance_count": int(self.reappearance_count),
            "reconnect_count": int(self.reconnect_count),
        }


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
    )
    tracker = IoUTracker(iou_threshold=cfg.tracker.iou_threshold, max_missed=cfg.tracker.max_missed)
    return ReidRuntime(
        detector=detector,
        extractor=extractor,
        gallery=gallery,
        tracker=tracker,
        confirm_hits=cfg.tracker.confirm_hits,
        new_identity_candidate_id=cfg.tracker.new_identity_candidate_id,
    )


def save_config_snapshot(path: Path, cfg: ReidRunConfig) -> None:
    snapshot = build_reid_config_snapshot(cfg)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
