from pathlib import Path
import time
import cv2

from pipeline.utils.logging import setup_logging
from pipeline.utils.paths import find_project_root, make_run_dir
from pipeline.utils.video import open_video, get_video_props, open_writer_avi_mjpg
from pipeline.detection.yolo import YoloDetector
from pipeline.reid.extractor import ReIDExtractor
from pipeline.reid.gallery import ReIDGallery, l2_normalize
from pipeline.tracking.iou import IoUTracker


def main() -> None:
    project_root = find_project_root(Path(__file__))

    input_video = project_root / "assets" / "videos" / "test.mp4"
    outputs_root = project_root / "outputs"
    run_dir = make_run_dir(outputs_root, prefix="runs")
    output_video = run_dir / "demo_reid_with_tracking.avi"

    logger = setup_logging(log_file=run_dir / "run.log")

    if not input_video.exists():
        raise FileNotFoundError(f"Put a video here: {input_video}")

    detector = YoloDetector(model_name="yolo26n.pt", conf=0.25, classes=[0])
    extractor = ReIDExtractor()
    gallery = ReIDGallery(sim_threshold=0.55, ema=0.8, update_threshold=0.60)
    tracker = IoUTracker(iou_threshold=0.3, max_missed=15)
    confirm_hits = 3

    logger.info(f"Input video:  {input_video}")
    logger.info(f"Run dir:      {run_dir}")
    logger.info(f"Output video: {output_video}")

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
        track_dets = tracker.update(dets)

        crops = []
        items = []
        for td in track_dets:
            x1, y1, x2, y2 = td.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            items.append((td.track_id, x1, y1, x2, y2, td.conf))

        used_person_ids = set()
        used_candidate_ids = set()

        if crops:
            feats = l2_normalize(extractor(crops))
            for i, (track_id, x1, y1, x2, y2, conf) in enumerate(items):
                emb = feats[i]

                pid = tracker.get_person_id(track_id)

                if pid is not None:
                    sim = gallery.similarity(pid, emb)
                    if gallery.should_update(sim):
                        gallery.update(pid, emb)
                    used_person_ids.add(pid)
                else:
                    forbidden = used_person_ids | used_candidate_ids
                    match = gallery.match(emb, forbidden_ids=forbidden, create_new=False)

                    cand_id = match.person_id
                    cand_sim = match.similarity

                    if cand_id == -1:
                        cand_id = 0
                        cand_sim = float("-inf")

                    if cand_id != 0:
                        confirmed = tracker.propose_person_id(track_id, cand_id, confirm_hits)
                        if confirmed is not None:
                            pid = confirmed
                            used_person_ids.add(pid)
                            if gallery.should_update(cand_sim):
                                gallery.update(pid, emb)
                        else:
                            used_candidate_ids.add(cand_id)
                    else:
                        confirmed = tracker.propose_person_id(track_id, 0, confirm_hits)
                        if confirmed is not None:
                            pid = gallery.add(emb)
                            tracker.set_person_id(track_id, pid)
                            used_person_ids.add(pid)
                        else:
                            used_candidate_ids.add(0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"tid {track_id} | id {pid} | det {conf:.2f}",
                    (x1, max(0, y1 - 6)),
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
                f"dets={len(dets)} | tracks_now={len(tracker.tracks())} | "
                f"tracks_updated={len(track_dets)} | gallery={len(gallery)} | fps={fps_proc:.2f}"
            )

    cap.release()
    out.release()
    logger.info(f"Finished {frame_idx} frames in {time.time() - start:.2f}s")
    logger.info(f"Saved output to: {output_video}")


if __name__ == "__main__":
    main()
