from pathlib import Path
import time
import cv2

from pipeline.utils.logging import setup_logging
from pipeline.utils.video import open_video, get_video_props, open_writer_avi_mjpg
from pipeline.detection.yolo import YoloDetector
from pipeline.reid.extractor import ReIDExtractor
from pipeline.reid.gallery import ReIDGallery, l2_normalize


def main() -> None:
    logger = setup_logging()

    project_root = Path(__file__).resolve().parents[2]
    input_video = project_root / "assets" / "videos" / "test.mp4"
    output_dir = project_root / "outputs" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "demo_reid_min.avi"

    detector = YoloDetector(model_name="yolo26n.pt", conf=0.25, classes=[0])
    extractor = ReIDExtractor()
    gallery = ReIDGallery(sim_threshold=0.55, ema=0.8)

    cap = open_video(input_video)
    props = get_video_props(cap)
    out = open_writer_avi_mjpg(output_video, props)

    log_every = 30
    start = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.predict(frame)

        crops = []
        boxes = []
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            boxes.append((x1, y1, x2, y2, d.conf))

        if crops:
            feats = l2_normalize(extractor(crops))
            for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
                match = gallery.match(feats[i])
                gallery.update(match.person_id, feats[i])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"id {match.person_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        out.write(frame)
        frame_idx += 1

        if frame_idx % log_every == 0:
            elapsed = time.time() - start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0

            logger.info(
                f"[frame {frame_idx}/{props.frame_count}] "
                f"dets={len(dets)} | "
                f"reid_crops={len(crops)} | "
                f"gallery={len(gallery)} | "
                f"fps={fps_proc:.2f}"
            )

    cap.release()
    out.release()
    logger.info(f"Finished {frame_idx} frames in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
