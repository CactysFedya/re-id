from pathlib import Path
import time

import cv2

from pipeline.utils.logging import setup_logging
from pipeline.utils.paths import find_project_root, make_run_dir
from pipeline.utils.video import get_video_props, open_output_writer, open_video
from pipeline.detection.yolo import YoloDetector


def main() -> None:
    project_root = find_project_root(Path(__file__))

    input_video = project_root / "assets" / "videos" / "test.mp4"
    outputs_root = project_root / "outputs" / "videos"
    run_dir = make_run_dir(outputs_root, prefix="detect")
    requested_output_video = run_dir / "demo_detect_yolo26.mp4"

    logger = setup_logging(log_file=run_dir / "run.log")

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
    logger.info(f"Run dir:      {run_dir}")

    cap = open_video(input_video)
    props = get_video_props(cap)
    logger.info(f"Video opened: {props.width}x{props.height}, fps={props.fps:.2f}, frames={props.frame_count}")

    video_writer = open_output_writer(requested_output_video, props)
    output_video = video_writer.path
    logger.info(f"Output video: {output_video}")
    logger.info(f"Output codec: {video_writer.codec}")
    if output_video != requested_output_video:
        logger.info(
            "Adjusted output container for compatibility: "
            f"{requested_output_video.name} -> {output_video.name}"
        )

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
                f"cls {d.cls} {d.conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        video_writer.writer.write(frame)
        frame_idx += 1

        if frame_idx % log_every == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            logger.info(f"Processed {frame_idx}/{props.frame_count} frames ({fps_proc:.2f} FPS)")

    cap.release()
    video_writer.writer.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    logger.info(f"Finished: frames={frame_idx}, time={total_time:.2f}s, avg_fps={avg_fps:.2f}")
    logger.info(f"Saved output to: {output_video}")


if __name__ == "__main__":
    main()
