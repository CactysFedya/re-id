from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass(frozen=True)
class VideoProps:
    fps: float
    width: int
    height: int
    frame_count: int


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def get_video_props(cap: cv2.VideoCapture) -> VideoProps:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoProps(fps=float(fps), width=width, height=height, frame_count=frame_count)


def open_writer_avi_mjpg(path: Path, props: VideoProps) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, props.fps, (props.width, props.height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open (codec/container issue)")
    return out
