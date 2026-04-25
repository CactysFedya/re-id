from __future__ import annotations

import argparse
import json
from pathlib import Path


MIN_VISIBILITY = 0.25
VALID_FLAG = 1
PEDESTRIAN_CLASS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pipeline metrics with MOT17 ground-truth aggregates for functional verification."
    )
    parser.add_argument(
        "result",
        type=Path,
        help="Path to metrics.json or to a run directory that contains metrics.json.",
    )
    parser.add_argument(
        "gt",
        type=Path,
        help="Path to MOT17 gt.txt.",
    )
    return parser.parse_args()


def resolve_metrics_path(candidate: Path) -> Path:
    candidate = Path(candidate)
    if candidate.is_dir():
        candidate = candidate / "metrics.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Metrics file not found: {candidate}")
    return candidate


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_gt(path: Path, frames_limit: int) -> dict[str, int | float]:
    unique_ids: set[int] = set()
    total_detections = 0
    max_frame = 0

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = line.strip().split(",")
            if len(row) < 9:
                continue

            frame = int(row[0])
            max_frame = max(max_frame, frame)
            if frame > frames_limit:
                continue

            object_id = int(row[1])
            valid_flag = int(float(row[6]))
            class_id = int(float(row[7]))
            visibility = float(row[8])

            if valid_flag != VALID_FLAG or class_id != PEDESTRIAN_CLASS or visibility < MIN_VISIBILITY:
                continue

            total_detections += 1
            unique_ids.add(object_id)

    frames_evaluated = min(frames_limit, max_frame) if max_frame > 0 else frames_limit
    avg_detections_per_frame = total_detections / frames_evaluated if frames_evaluated > 0 else 0.0
    return {
        "frames_evaluated": int(frames_evaluated),
        "gt_total_detections": int(total_detections),
        "gt_unique_ids": int(len(unique_ids)),
        "gt_avg_detections_per_frame": round(avg_detections_per_frame, 4),
    }


def ratio(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return round(value / reference, 4)


def interpret_global_ids(diff: int) -> str:
    if diff > 0:
        return "Система создала больше глобальных идентификаторов, чем отмечено уникальных объектов в разметке; это может указывать на дробление идентичностей."
    if diff < 0:
        return "Система создала меньше глобальных идентификаторов, чем отмечено уникальных объектов в разметке; это может указывать на слияние разных объектов или пропуск части наблюдений."
    return "Число глобальных идентификаторов совпало с числом уникальных объектов в разметке."


def interpret_detections(diff: int) -> str:
    if diff > 0:
        return "Система сформировала больше детекций, чем содержится в отфильтрованной разметке; возможны ложные срабатывания или более чувствительный режим обнаружения."
    if diff < 0:
        return "Система сформировала меньше детекций, чем содержится в отфильтрованной разметке; возможны пропуски наблюдений или слишком строгий порог обнаружения."
    return "Число детекций системы совпало с числом детекций в отфильтрованной разметке."


def build_summary(metrics_path: Path, gt_path: Path) -> dict:
    metrics = load_metrics(metrics_path)
    frames_processed = int(metrics.get("frames_processed", 0))
    gt_summary = summarize_gt(gt_path, frames_processed)

    system_total_detections = int(metrics.get("total_detections", 0))
    system_total_global_ids = int(metrics.get("total_global_ids_created", 0))
    gt_total_detections = int(gt_summary["gt_total_detections"])
    gt_unique_ids = int(gt_summary["gt_unique_ids"])

    detection_difference = system_total_detections - gt_total_detections
    global_id_difference = system_total_global_ids - gt_unique_ids

    return {
        "result_path": str(metrics_path),
        "gt_path": str(gt_path),
        "gt_filter": {
            "valid_flag": VALID_FLAG,
            "class_id": PEDESTRIAN_CLASS,
            "min_visibility": MIN_VISIBILITY,
        },
        "frames_processed": frames_processed,
        "frames_evaluated": int(gt_summary["frames_evaluated"]),
        "gt_total_detections": gt_total_detections,
        "system_total_detections": system_total_detections,
        "detection_difference": int(detection_difference),
        "detection_ratio": ratio(system_total_detections, gt_total_detections),
        "gt_avg_detections_per_frame": gt_summary["gt_avg_detections_per_frame"],
        "system_avg_detections_per_frame": round(float(metrics.get("avg_detections_per_frame", 0.0)), 4),
        "gt_unique_ids": gt_unique_ids,
        "system_global_ids": system_total_global_ids,
        "global_id_difference": int(global_id_difference),
        "global_id_ratio": ratio(system_total_global_ids, gt_unique_ids),
        "reappearance_count": int(metrics.get("reappearance_count", 0)),
        "cross_session_reappearance_count": int(metrics.get("cross_session_reappearance_count", 0)),
        "avg_fps": round(float(metrics.get("avg_fps", 0.0)), 4),
        "source_fps": round(float(metrics.get("source_fps", 0.0)), 4),
        "processing_to_source_fps_ratio": round(float(metrics.get("processing_to_source_fps_ratio", 0.0)), 4),
        "global_id_interpretation": interpret_global_ids(global_id_difference),
        "detection_interpretation": interpret_detections(detection_difference),
    }


def main() -> None:
    args = parse_args()
    metrics_path = resolve_metrics_path(args.result)
    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    summary = build_summary(metrics_path, gt_path)
    output_path = metrics_path.parent / "mot17_eval_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
