from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit
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


def open_video_source(
    *,
    source_type: str,
    uri: Optional[str] = None,
    device_index: Optional[int] = None,
    buffer_size: int = 1,
) -> cv2.VideoCapture:
    source_type = source_type.strip().lower()

    if source_type == "file":
        if not uri:
            raise ValueError("File source requires uri")
        return open_video(Path(uri))

    if source_type in {"usb", "camera"}:
        cap = cv2.VideoCapture(0 if device_index is None else int(device_index))
    elif source_type in {"rtsp", "http", "https", "mjpeg"}:
        if not uri:
            raise ValueError(f"{source_type} source requires uri")
        if source_type in {"http", "https", "mjpeg"}:
            return _open_http_source(uri, buffer_size=buffer_size)
        cap = cv2.VideoCapture(uri)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    if not cap.isOpened():
        target = device_index if source_type in {"usb", "camera"} else uri
        raise RuntimeError(f"Cannot open source {source_type}: {target}")

    if buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

    return cap


def _open_http_source(uri: str, buffer_size: int) -> cv2.VideoCapture:
    attempted_urls = []
    for candidate in _http_candidates(uri):
        attempted_urls.append(candidate)
        cap = cv2.VideoCapture(candidate)
        if cap.isOpened():
            if buffer_size > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
            return cap
        cap.release()

    attempted_text = ", ".join(attempted_urls)
    raise RuntimeError(
        "Cannot open HTTP video source. "
        f"Tried: {attempted_text}. "
        "If this is a phone/IP webcam, use the direct stream URL such as /video or /mjpeg."
    )


def _http_candidates(uri: str) -> list[str]:
    candidates = [uri]
    parts = urlsplit(uri)
    normalized_path = parts.path.rstrip("/")

    if normalized_path:
        return candidates

    for suffix in ("/video", "/mjpeg", "/video.mjpg", "/mjpg/video.mjpg"):
        candidates.append(urlunsplit((parts.scheme, parts.netloc, suffix, parts.query, parts.fragment)))

    return candidates


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
        raise RuntimeError("VideoWriter failed to open")
    return out
