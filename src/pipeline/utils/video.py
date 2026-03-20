from dataclasses import dataclass
from pathlib import Path
import shutil
import threading
from typing import Optional
from urllib.parse import urlsplit, urlunsplit
import cv2


@dataclass(frozen=True)
class VideoProps:
    fps: float
    width: int
    height: int
    frame_count: int


class LatestFrameCapture:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._reader_loop, name="latest-frame-capture", daemon=True)
        self._latest_frame = None
        self._read_failed = False
        self._stopped = False
        self._overwritten_count = 0

    def start(self) -> None:
        self._thread.start()

    def _reader_loop(self) -> None:
        while True:
            with self._cond:
                if self._stopped:
                    return

            ok, frame = self._cap.read()

            with self._cond:
                if self._stopped:
                    return
                if not ok:
                    self._read_failed = True
                    self._cond.notify_all()
                    return
                if self._latest_frame is not None:
                    self._overwritten_count += 1
                self._latest_frame = frame
                self._cond.notify_all()

    def read(self, timeout_s: float | None = None):
        with self._cond:
            if timeout_s is None:
                while self._latest_frame is None and not self._read_failed and not self._stopped:
                    self._cond.wait()
            else:
                self._cond.wait_for(
                    lambda: self._latest_frame is not None or self._read_failed or self._stopped,
                    timeout=timeout_s,
                )

            if self._latest_frame is not None:
                frame = self._latest_frame
                self._latest_frame = None
                return True, frame

            return False, None

    def consume_overwritten_count(self) -> int:
        with self._cond:
            count = self._overwritten_count
            self._overwritten_count = 0
            return count

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


class TimelineVideoRecorder:
    def __init__(self, frames_dir: Path):
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_paths: list[Path] = []
        self._timestamps_s: list[float] = []

    def add_frame(self, frame, timestamp_s: float) -> None:
        frame_path = self.frames_dir / f"{len(self._frame_paths):06d}.jpg"
        ok = cv2.imwrite(str(frame_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to save frame to {frame_path}")
        self._frame_paths.append(frame_path)
        self._timestamps_s.append(float(timestamp_s))

    def finalize(self, output_path: Path, props: VideoProps, total_duration_s: float) -> float:
        if not self._frame_paths:
            return 0.0

        timeline_fps = props.fps if props.fps > 0 else 30.0
        writer = open_writer_avi_mjpg(
            output_path,
            VideoProps(
                fps=float(timeline_fps),
                width=props.width,
                height=props.height,
                frame_count=len(self._frame_paths),
            ),
        )
        try:
            for idx, frame_path in enumerate(self._frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue

                current_ts = self._timestamps_s[idx]
                if idx + 1 < len(self._timestamps_s):
                    next_ts = self._timestamps_s[idx + 1]
                else:
                    next_ts = max(total_duration_s, current_ts + 1.0 / timeline_fps)

                duration_s = max(1.0 / timeline_fps, next_ts - current_ts)
                repeat = max(1, int(round(duration_s * timeline_fps)))
                for _ in range(repeat):
                    writer.write(frame)
        finally:
            writer.release()
            shutil.rmtree(self.frames_dir, ignore_errors=True)

        return float(timeline_fps)


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
