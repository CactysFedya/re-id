from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pipeline.detection.yolo import Detection


BBox = Tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Track:
    track_id: int
    bbox: BBox
    conf: float
    missed: int = 0
    person_id: Optional[int] = None


@dataclass(frozen=True)
class TrackDet:
    track_id: int
    bbox: BBox
    conf: float


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._next_track_id = 1
        self._tracks: Dict[int, Track] = {}

    def tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def update(self, detections: List[Detection]) -> List[TrackDet]:
        for t in self._tracks.values():
            t.missed += 1

        det_bboxes = [d.xyxy for d in detections]

        pairs: List[Tuple[int, int, float]] = []
        for tid, tr in self._tracks.items():
            for j, bb in enumerate(det_bboxes):
                s = iou(tr.bbox, bb)
                if s >= self.iou_threshold:
                    pairs.append((tid, j, s))

        pairs.sort(key=lambda x: x[2], reverse=True)
        used_tracks = set()
        used_dets = set()
        assignment: Dict[int, int] = {}

        for tid, j, s in pairs:
            if tid in used_tracks or j in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(j)
            assignment[tid] = j

        for tid, j in assignment.items():
            d = detections[j]
            tr = self._tracks[tid]
            tr.bbox = d.xyxy
            tr.conf = d.conf
            tr.missed = 0

        for j, d in enumerate(detections):
            if j in used_dets:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = Track(track_id=tid, bbox=d.xyxy, conf=d.conf, missed=0)

        dead = [tid for tid, t in self._tracks.items() if t.missed > self.max_missed]
        for tid in dead:
            del self._tracks[tid]

        out: List[TrackDet] = []
        for tid, t in self._tracks.items():
            if t.missed == 0:
                out.append(TrackDet(track_id=tid, bbox=t.bbox, conf=t.conf))
        return out

    def set_person_id(self, track_id: int, person_id: int) -> None:
        tr = self._tracks.get(track_id)
        if tr is not None:
            tr.person_id = person_id

    def get_person_id(self, track_id: int) -> Optional[int]:
        tr = self._tracks.get(track_id)
        return tr.person_id if tr is not None else None
