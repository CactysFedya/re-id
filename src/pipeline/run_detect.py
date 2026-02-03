from pathlib import Path
import time

import cv2

from reid.utils.logging import setup_logging
from reid.utils.paths import find_project_root
from reid.io.video import open_video, get_video_props, open_writer_avi_mjpg
from reid.detection.yolo import YoloDetector


def main() -> None:
    logger = setup_logging()

    project_root = find_project_root(Path(__file__))
    input_video = project_root / "assets" / "videos" / "test.mp4"
    output_dir = project_root / "outputs" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "demo_detect_yolo26.avi"

    if not input_video.exists():
        raise FileNotFoundError(f"Put a video here: {input_video}")

    local_weights = project_root / "models" / "detectors" / "yolo26n.pt"

    detector = YoloDetector(
        model_name="yolo26n.pt",
        weights_path=local_weights,
        conf=0.25,
        classes=[0],
    )

    logger.info(f"Input video:  {input_video}")
    logger.info(f"Output video: {output_video}")

    cap = open_video(input_video)
    props = get_video_props(cap)
    logger.info(f"Video opened: {props.width}x{props.height}, fps={props.fps:.2f}, frames={props.frame_count}")

    out = open_writer_avi_mjpg(output_video, props)

    start_time = time.time()
    frame_idx = 0
    log_every = 30

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.predict(frame)

        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"person {d.conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out.write(frame)
        frame_idx += 1

        if frame_idx % log_every == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            logger.info(f"Processed {frame_idx}/{props.frame_count} frames ({fps_proc:.2f} FPS)")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    logger.info(f"Finished: frames={frame_idx}, time={total_time:.2f}s, avg_fps={avg_fps:.2f}")
    logger.info(f"Saved output to: {output_video}")


if __name__ == "__main__":
    main()
