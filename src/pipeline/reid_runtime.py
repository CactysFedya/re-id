from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import cv2

from pipeline.reid.gallery import ReIDGallery, l2_normalize
from pipeline.tracking.iou import IoUTracker
from pipeline.utils.performance import StageTimer


_NEW_IDENTITY_CANDIDATE_ID = 0


@dataclass
class ReidRuntime:
    detector: YoloDetector
    extractor: ReIDExtractor
    gallery: ReIDGallery
    tracker: IoUTracker
    confirm_hits: int
    perf: StageTimer = field(default_factory=StageTimer)
    frame_idx: int = 0
    total_frame_time_s: float = 0.0
    reappearance_count: int = 0
    skipped_frame_count: int = 0
    reconnect_count: int = 0
    last_gallery_autosave_monotonic_s: float | None = None
    identity_last_track: dict[int, int] = field(default_factory=dict)
    initial_gallery_ids: set[int] = field(default_factory=set)
    cross_session_reappearance_count: int = 0
    draw_color: tuple[int, int, int] = (0, 255, 0)
    draw_box_thickness: int = 2
    draw_font_scale: float = 0.6
    draw_text_thickness: int = 2

    def set_gallery(self, gallery: ReIDGallery) -> None:
        self.gallery = gallery
        self.initial_gallery_ids = gallery.identity_ids()
        self.identity_last_track.clear()
        self.reappearance_count = 0
        self.cross_session_reappearance_count = 0

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
            items.append((td.track_id, td.cls, x1, y1, x2, y2, td.conf))
        self.perf.add("crop", time.perf_counter() - crop_started, count=len(crops))

        used_identity_ids = set()
        used_candidate_ids = set()
        protected_identity_ids = {
            track.identity_id
            for track in self.tracker.tracks()
            if track.identity_id is not None
        }

        if crops:
            feature_started = time.perf_counter()
            feats = l2_normalize(self.extractor(crops))
            self.perf.add("reid", time.perf_counter() - feature_started, count=len(crops))

            draw_started = time.perf_counter()
            for i, (track_id, cls_id, x1, y1, x2, y2, conf) in enumerate(items):
                emb = feats[i]
                identity_id = self.tracker.get_identity_id(track_id)

                if identity_id is not None:
                    self.gallery.note_seen(identity_id)
                    sim = self.gallery.similarity(identity_id, emb)
                    if self.gallery.should_update(sim):
                        self.gallery.update(identity_id, emb)
                    used_identity_ids.add(identity_id)
                else:
                    forbidden = used_identity_ids | used_candidate_ids
                    match = self.gallery.match(emb, forbidden_ids=forbidden, create_new=False, label=cls_id)
                    cand_id = match.identity_id
                    cand_sim = match.similarity

                    if cand_id == -1:
                        cand_id = _NEW_IDENTITY_CANDIDATE_ID
                        cand_sim = float("-inf")

                    if cand_id != _NEW_IDENTITY_CANDIDATE_ID:
                        confirmed = self.tracker.propose_identity_id(track_id, cand_id, self.confirm_hits)
                        if confirmed is not None:
                            identity_id = confirmed
                            self.gallery.note_seen(identity_id)
                            used_identity_ids.add(identity_id)
                            if self.gallery.should_update(cand_sim):
                                self.gallery.update(identity_id, emb)
                            prev_track_id = self.identity_last_track.get(identity_id)
                            if prev_track_id is None and identity_id in self.initial_gallery_ids:
                                self.reappearance_count += 1
                                self.cross_session_reappearance_count += 1
                            elif prev_track_id is not None and prev_track_id != track_id:
                                self.reappearance_count += 1
                            self.identity_last_track[identity_id] = track_id
                        else:
                            used_candidate_ids.add(cand_id)
                    else:
                        confirmed = self.tracker.propose_identity_id(
                            track_id,
                            _NEW_IDENTITY_CANDIDATE_ID,
                            self.confirm_hits,
                        )
                        if confirmed is not None:
                            identity_id = self.gallery.add(emb, label=cls_id, protected_ids=protected_identity_ids)
                            self.tracker.set_identity_id(track_id, identity_id)
                            used_identity_ids.add(identity_id)
                            self.identity_last_track[identity_id] = track_id
                        else:
                            used_candidate_ids.add(_NEW_IDENTITY_CANDIDATE_ID)

                cv2.rectangle(frame, (x1, y1), (x2, y2), self.draw_color, self.draw_box_thickness)
                cv2.putText(
                    frame,
                    f"cls {cls_id} | tid {track_id} | id {identity_id} | det {conf:.2f}",
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
        total_detections = self.perf.total("detect")
        total_tracked_detections = self.perf.total("track")
        total_crops_encoded = self.perf.total("crop")
        avg_detections_per_frame = total_detections / self.frame_idx if self.frame_idx > 0 else 0.0
        avg_crops_per_frame = total_crops_encoded / self.frame_idx if self.frame_idx > 0 else 0.0
        crop_utilization_ratio = total_crops_encoded / total_detections if total_detections > 0 else 0.0
        total_reid_s = self.perf.totals_s.get("reid", 0.0)
        avg_reid_ms_per_crop = total_reid_s / total_crops_encoded * 1000.0 if total_crops_encoded > 0 else 0.0
        return {
            "frames_processed": self.frame_idx,
            "frames_skipped": int(self.skipped_frame_count),
            "avg_fps": round(avg_fps, 4),
            "avg_total_ms": round(avg_total_ms, 4),
            "avg_detections_per_frame": round(avg_detections_per_frame, 4),
            "avg_crops_per_frame": round(avg_crops_per_frame, 4),
            "crop_utilization_ratio": round(crop_utilization_ratio, 4),
            "avg_reid_ms_per_crop": round(avg_reid_ms_per_crop, 4),
            **self.perf.summary(self.frame_idx, exclude={"loop"}),
            "total_detections": total_detections,
            "total_tracked_detections": total_tracked_detections,
            "total_crops_encoded": total_crops_encoded,
            "total_global_ids_created": int(self.gallery.total_ids_created()),
            "total_autosaves": self.perf.total("autosave"),
            "total_gallery_evictions": int(self.gallery.total_evictions()),
            "reappearance_count": int(self.reappearance_count),
            "cross_session_reappearance_count": int(self.cross_session_reappearance_count),
            "reconnect_count": int(self.reconnect_count),
        }
