from pathlib import Path
import sys
import logging
import time

import cv2
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    start_time = time.time()
    logger.info("Starting YOLO26 video detection")

    project_root = Path(__file__).resolve().parents[2]

    input_video = project_root / "assets" / "videos" / "test.mp4"
    output_dir = project_root / "outputs" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "demo_detect_yolo26.avi"

    if not input_video.exists():
        raise FileNotFoundError(f"Put a video here: {input_video}")

    logger.info(f"Input video: {input_video}")
    logger.info(f"Output video: {output_video}")

    logger.info("Loading YOLO26 model (yolo26n.pt)")
    model = YOLO("yolo26n.pt")
    logger.info("Model loaded")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video opened: {w}x{h}, fps={fps:.2f}, frames={total_frames}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(output_video), fourcc, float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open (try a different codec)")

    conf_thres = 0.25
    log_every_n_frames = 30

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.info("End of video stream reached")
            break

        result = model.predict(
            frame,
            verbose=False,
            conf=conf_thres,
            classes=[0],
        )[0]

        for b in result.boxes:
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"person {conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out.write(frame)
        frame_idx += 1

        if frame_idx % log_every_n_frames == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"Processed {frame_idx}/{total_frames} frames "
                f"({fps_proc:.2f} FPS)"
            )

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

    logger.info(
        f"Finished processing {frame_idx} frames in {total_time:.2f}s "
        f"(avg {avg_fps:.2f} FPS)"
    )
    logger.info(f"Saved output to: {output_video}")


if __name__ == "__main__":
    main()
