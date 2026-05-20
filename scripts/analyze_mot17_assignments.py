from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MIN_VISIBILITY = 0.25
VALID_FLAG = 1
PEDESTRIAN_CLASS = 1
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_REAPPEARANCE_GAP = 15


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class GroundTruth:
    frame: int
    gt_id: int
    box: Box
    visibility: float


@dataclass(frozen=True)
class Assignment:
    frame: int
    track_id: int | None
    global_id: int | None
    box: Box
    conf: float | None


@dataclass(frozen=True)
class Match:
    frame: int
    gt_id: int
    global_id: int | None
    track_id: int | None
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match pipeline assignments.csv with MOT17 gt.txt frame by frame and "
            "compute detection and ReID-oriented summary metrics."
        )
    )
    parser.add_argument(
        "assignments",
        type=Path,
        help="Path to assignments.csv or to a run directory that contains assignments.csv.",
    )
    parser.add_argument("gt", type=Path, help="Path to MOT17 gt.txt.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Minimum IoU for matching assignment boxes to GT boxes. Default: {DEFAULT_IOU_THRESHOLD}.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=MIN_VISIBILITY,
        help=f"Minimum MOT17 visibility for GT rows. Default: {MIN_VISIBILITY}.",
    )
    parser.add_argument(
        "--valid-flag",
        type=int,
        default=VALID_FLAG,
        help=f"MOT17 valid/conf flag to keep. Default: {VALID_FLAG}.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=PEDESTRIAN_CLASS,
        help=f"MOT17 class id to keep. Default: {PEDESTRIAN_CLASS} (pedestrian).",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Optional frame limit for comparing only the first N frames.",
    )
    parser.add_argument(
        "--reappearance-gap",
        type=int,
        default=DEFAULT_REAPPEARANCE_GAP,
        help=(
            "Minimum gap in frames between matched GT observations to count a reappearance event. "
            f"Default: {DEFAULT_REAPPEARANCE_GAP}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <run-dir>/mot17_assignment_analysis.json.",
    )
    parser.add_argument(
        "--write-tables",
        action="store_true",
        help="Also write gt_to_global_id.csv and global_id_to_gt.csv next to the JSON summary.",
    )
    return parser.parse_args()


def resolve_assignments_path(candidate: Path) -> Path:
    candidate = Path(candidate)
    if candidate.is_dir():
        candidate = candidate / "assignments.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Assignments file not found: {candidate}")
    return candidate


def default_output_path(assignments_path: Path) -> Path:
    return assignments_path.parent / "mot17_assignment_analysis.json"


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def load_assignments(path: Path, max_frame: int | None) -> dict[int, list[Assignment]]:
    by_frame: dict[int, list[Assignment]] = defaultdict(list)
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"frame", "track_id", "global_id", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Assignments file is missing columns: {sorted(missing)}")

        for row in reader:
            frame = int(row["frame"])
            if max_frame is not None and frame > max_frame:
                continue
            by_frame[frame].append(
                Assignment(
                    frame=frame,
                    track_id=parse_optional_int(row.get("track_id")),
                    global_id=parse_optional_int(row.get("global_id")),
                    box=Box(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        w=float(row["w"]),
                        h=float(row["h"]),
                    ),
                    conf=parse_optional_float(row.get("conf")),
                )
            )
    return dict(by_frame)


def load_gt(
    path: Path,
    *,
    max_frame: int | None,
    valid_flag: int,
    class_id: int,
    min_visibility: float,
) -> dict[int, list[GroundTruth]]:
    by_frame: dict[int, list[GroundTruth]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = line.strip().split(",")
            if len(row) < 9:
                continue

            frame = int(row[0])
            if max_frame is not None and frame > max_frame:
                continue

            row_valid_flag = int(float(row[6]))
            row_class_id = int(float(row[7]))
            visibility = float(row[8])
            if row_valid_flag != valid_flag or row_class_id != class_id or visibility < min_visibility:
                continue

            by_frame[frame].append(
                GroundTruth(
                    frame=frame,
                    gt_id=int(row[1]),
                    box=Box(x=float(row[2]), y=float(row[3]), w=float(row[4]), h=float(row[5])),
                    visibility=visibility,
                )
            )
    return dict(by_frame)


def iou(a: Box, b: Box) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union_area = a.w * a.h + b.w * b.h - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def greedy_match_frame(
    gt_items: list[GroundTruth],
    assignment_items: list[Assignment],
    iou_threshold: float,
) -> tuple[list[Match], int, int]:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_item in enumerate(gt_items):
        for assignment_idx, assignment in enumerate(assignment_items):
            score = iou(gt_item.box, assignment.box)
            if score >= iou_threshold:
                candidates.append((score, gt_idx, assignment_idx))

    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_assignments: set[int] = set()
    matches: list[Match] = []

    for score, gt_idx, assignment_idx in candidates:
        if gt_idx in used_gt or assignment_idx in used_assignments:
            continue
        used_gt.add(gt_idx)
        used_assignments.add(assignment_idx)
        gt_item = gt_items[gt_idx]
        assignment = assignment_items[assignment_idx]
        matches.append(
            Match(
                frame=gt_item.frame,
                gt_id=gt_item.gt_id,
                global_id=assignment.global_id,
                track_id=assignment.track_id,
                iou=score,
            )
        )

    return matches, len(gt_items) - len(used_gt), len(assignment_items) - len(used_assignments)


def match_all_frames(
    gt_by_frame: dict[int, list[GroundTruth]],
    assignments_by_frame: dict[int, list[Assignment]],
    iou_threshold: float,
) -> tuple[list[Match], int, int]:
    frames = sorted(set(gt_by_frame) | set(assignments_by_frame))
    all_matches: list[Match] = []
    unmatched_gt = 0
    unmatched_assignments = 0

    for frame in frames:
        matches, frame_unmatched_gt, frame_unmatched_assignments = greedy_match_frame(
            gt_by_frame.get(frame, []),
            assignments_by_frame.get(frame, []),
            iou_threshold,
        )
        all_matches.extend(matches)
        unmatched_gt += frame_unmatched_gt
        unmatched_assignments += frame_unmatched_assignments

    return all_matches, unmatched_gt, unmatched_assignments


def round_ratio(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return round(value / reference, 4)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return 0.0
    return round(2.0 * precision * recall / denom, 4)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_pair_counters(matches: list[Match]) -> tuple[Counter[tuple[int, int]], Counter[tuple[int, int]]]:
    gt_global_pairs: Counter[tuple[int, int]] = Counter()
    global_gt_pairs: Counter[tuple[int, int]] = Counter()
    for match in matches:
        if match.global_id is None:
            continue
        gt_global_pairs[(match.gt_id, match.global_id)] += 1
        global_gt_pairs[(match.global_id, match.gt_id)] += 1
    return gt_global_pairs, global_gt_pairs


def count_id_switches(matches: list[Match]) -> int:
    by_gt: dict[int, list[Match]] = defaultdict(list)
    for match in matches:
        if match.global_id is not None:
            by_gt[match.gt_id].append(match)

    switches = 0
    for gt_matches in by_gt.values():
        gt_matches.sort(key=lambda item: item.frame)
        previous_global_id = None
        for match in gt_matches:
            if previous_global_id is not None and match.global_id != previous_global_id:
                switches += 1
            previous_global_id = match.global_id
    return switches


def analyze_reappearances(matches: list[Match], gap: int) -> dict[str, int | float | None]:
    by_gt: dict[int, list[Match]] = defaultdict(list)
    for match in matches:
        if match.global_id is not None:
            by_gt[match.gt_id].append(match)

    events = 0
    same_global_id = 0
    changed_global_id = 0

    for gt_matches in by_gt.values():
        gt_matches.sort(key=lambda item: item.frame)
        previous = None
        for match in gt_matches:
            if previous is not None and match.frame - previous.frame > gap:
                events += 1
                if match.global_id == previous.global_id:
                    same_global_id += 1
                else:
                    changed_global_id += 1
            previous = match

    return {
        "gap_threshold_frames": int(gap),
        "events": int(events),
        "same_global_id": int(same_global_id),
        "changed_global_id": int(changed_global_id),
        "success_ratio": round_ratio(same_global_id, events),
    }


def build_identity_summary(matches: list[Match]) -> dict:
    gt_global_pairs, global_gt_pairs = build_pair_counters(matches)

    gt_to_global: dict[int, Counter[int]] = defaultdict(Counter)
    global_to_gt: dict[int, Counter[int]] = defaultdict(Counter)
    for (gt_id, global_id), count in gt_global_pairs.items():
        gt_to_global[gt_id][global_id] = count
    for (global_id, gt_id), count in global_gt_pairs.items():
        global_to_gt[global_id][gt_id] = count

    gt_fragment_counts = [len(counter) for counter in gt_to_global.values()]
    global_merge_counts = [len(counter) for counter in global_to_gt.values()]

    global_purities = []
    weighted_purity_num = 0
    weighted_purity_den = 0
    for counter in global_to_gt.values():
        total = sum(counter.values())
        dominant = max(counter.values()) if counter else 0
        if total > 0:
            global_purities.append(dominant / total)
            weighted_purity_num += dominant
            weighted_purity_den += total

    matched_with_global = sum(1 for match in matches if match.global_id is not None)
    matched_without_global = len(matches) - matched_with_global

    return {
        "matched_with_global_id": int(matched_with_global),
        "matched_without_global_id": int(matched_without_global),
        "gt_ids_matched_with_global_id": int(len(gt_to_global)),
        "system_global_ids_matched_to_gt": int(len(global_to_gt)),
        "fragmented_gt_ids": int(sum(1 for count in gt_fragment_counts if count > 1)),
        "avg_global_ids_per_gt_id": round(mean(gt_fragment_counts), 4),
        "max_global_ids_per_gt_id": int(max(gt_fragment_counts, default=0)),
        "merged_global_ids": int(sum(1 for count in global_merge_counts if count > 1)),
        "avg_gt_ids_per_global_id": round(mean(global_merge_counts), 4),
        "max_gt_ids_per_global_id": int(max(global_merge_counts, default=0)),
        "mean_global_id_purity": round(mean(global_purities), 4),
        "weighted_global_id_purity": round_ratio(weighted_purity_num, weighted_purity_den),
        "id_switches": int(count_id_switches(matches)),
    }


def top_mapping_rows(pair_counts: Counter[tuple[int, int]], limit: int = 20) -> list[dict[str, int]]:
    rows = []
    for (first_id, second_id), count in pair_counts.most_common(limit):
        rows.append({"first_id": int(first_id), "second_id": int(second_id), "matches": int(count)})
    return rows


def build_summary(
    *,
    assignments_path: Path,
    gt_path: Path,
    gt_by_frame: dict[int, list[GroundTruth]],
    assignments_by_frame: dict[int, list[Assignment]],
    matches: list[Match],
    unmatched_gt: int,
    unmatched_assignments: int,
    iou_threshold: float,
    min_visibility: float,
    valid_flag: int,
    class_id: int,
    reappearance_gap: int,
) -> dict:
    total_gt = sum(len(items) for items in gt_by_frame.values())
    total_assignments = sum(len(items) for items in assignments_by_frame.values())
    matched_count = len(matches)
    frames_evaluated = len(set(gt_by_frame) | set(assignments_by_frame))
    gt_ids_total = len({item.gt_id for items in gt_by_frame.values() for item in items})
    assignment_global_ids_total = len(
        {
            item.global_id
            for items in assignments_by_frame.values()
            for item in items
            if item.global_id is not None
        }
    )
    assignment_track_ids_total = len(
        {
            item.track_id
            for items in assignments_by_frame.values()
            for item in items
            if item.track_id is not None
        }
    )
    gt_global_pairs, global_gt_pairs = build_pair_counters(matches)
    recall = round_ratio(matched_count, total_gt)
    precision = round_ratio(matched_count, total_assignments)

    return {
        "assignments_path": str(assignments_path),
        "gt_path": str(gt_path),
        "parameters": {
            "iou_threshold": float(iou_threshold),
            "gt_valid_flag": int(valid_flag),
            "gt_class_id": int(class_id),
            "gt_min_visibility": float(min_visibility),
            "reappearance_gap_frames": int(reappearance_gap),
        },
        "detection": {
            "frames_evaluated": int(frames_evaluated),
            "gt_detections": int(total_gt),
            "system_assignments": int(total_assignments),
            "matched_detections": int(matched_count),
            "unmatched_gt": int(unmatched_gt),
            "unmatched_system_assignments": int(unmatched_assignments),
            "recall": recall,
            "precision": precision,
            "f1": f1_score(precision, recall),
            "avg_iou_matched": round(mean(match.iou for match in matches), 4),
        },
        "identity": {
            "gt_unique_ids": int(gt_ids_total),
            "system_global_ids_in_assignments": int(assignment_global_ids_total),
            "system_track_ids_in_assignments": int(assignment_track_ids_total),
            **build_identity_summary(matches),
            "reappearance_after_gap": analyze_reappearances(matches, reappearance_gap),
        },
        "top_gt_to_global_id_pairs": [
            {
                "gt_id": row["first_id"],
                "global_id": row["second_id"],
                "matches": row["matches"],
            }
            for row in top_mapping_rows(gt_global_pairs)
        ],
        "top_global_id_to_gt_pairs": [
            {
                "global_id": row["first_id"],
                "gt_id": row["second_id"],
                "matches": row["matches"],
            }
            for row in top_mapping_rows(global_gt_pairs)
        ],
    }


def write_mapping_table(path: Path, first_name: str, second_name: str, pair_counts: Counter[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[first_name, second_name, "matches"])
        writer.writeheader()
        for (first_id, second_id), count in pair_counts.most_common():
            writer.writerow({first_name: first_id, second_name: second_id, "matches": count})


def write_gt_summary_table(path: Path, matches: list[Match]) -> None:
    by_gt: dict[int, Counter[int]] = defaultdict(Counter)
    matched_frames: dict[int, set[int]] = defaultdict(set)
    for match in matches:
        matched_frames[match.gt_id].add(match.frame)
        if match.global_id is not None:
            by_gt[match.gt_id][match.global_id] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "gt_id",
                "matched_frames",
                "assigned_global_ids",
                "dominant_global_id",
                "dominant_matches",
                "fragmented",
            ],
        )
        writer.writeheader()
        for gt_id in sorted(matched_frames):
            counter = by_gt.get(gt_id, Counter())
            dominant_global_id = None
            dominant_matches = 0
            if counter:
                dominant_global_id, dominant_matches = counter.most_common(1)[0]
            writer.writerow(
                {
                    "gt_id": gt_id,
                    "matched_frames": len(matched_frames[gt_id]),
                    "assigned_global_ids": len(counter),
                    "dominant_global_id": "" if dominant_global_id is None else dominant_global_id,
                    "dominant_matches": dominant_matches,
                    "fragmented": int(len(counter) > 1),
                }
            )


def write_global_summary_table(path: Path, matches: list[Match]) -> None:
    by_global: dict[int, Counter[int]] = defaultdict(Counter)
    for match in matches:
        if match.global_id is not None:
            by_global[match.global_id][match.gt_id] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "global_id",
                "matched_gt_ids",
                "dominant_gt_id",
                "dominant_matches",
                "total_matches",
                "purity",
                "merged",
            ],
        )
        writer.writeheader()
        for global_id in sorted(by_global):
            counter = by_global[global_id]
            dominant_gt_id, dominant_matches = counter.most_common(1)[0]
            total_matches = sum(counter.values())
            writer.writerow(
                {
                    "global_id": global_id,
                    "matched_gt_ids": len(counter),
                    "dominant_gt_id": dominant_gt_id,
                    "dominant_matches": dominant_matches,
                    "total_matches": total_matches,
                    "purity": round_ratio(dominant_matches, total_matches),
                    "merged": int(len(counter) > 1),
                }
            )


def main() -> None:
    args = parse_args()
    assignments_path = resolve_assignments_path(args.assignments)
    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_by_frame = load_gt(
        gt_path,
        max_frame=args.max_frame,
        valid_flag=args.valid_flag,
        class_id=args.class_id,
        min_visibility=args.min_visibility,
    )
    assignments_by_frame = load_assignments(assignments_path, max_frame=args.max_frame)
    matches, unmatched_gt, unmatched_assignments = match_all_frames(
        gt_by_frame,
        assignments_by_frame,
        args.iou_threshold,
    )

    summary = build_summary(
        assignments_path=assignments_path,
        gt_path=gt_path,
        gt_by_frame=gt_by_frame,
        assignments_by_frame=assignments_by_frame,
        matches=matches,
        unmatched_gt=unmatched_gt,
        unmatched_assignments=unmatched_assignments,
        iou_threshold=args.iou_threshold,
        min_visibility=args.min_visibility,
        valid_flag=args.valid_flag,
        class_id=args.class_id,
        reappearance_gap=args.reappearance_gap,
    )

    output_path = args.output or default_output_path(assignments_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_tables:
        gt_global_pairs, global_gt_pairs = build_pair_counters(matches)
        write_mapping_table(output_path.parent / "gt_to_global_id.csv", "gt_id", "global_id", gt_global_pairs)
        write_mapping_table(output_path.parent / "global_id_to_gt.csv", "global_id", "gt_id", global_gt_pairs)
        write_gt_summary_table(output_path.parent / "gt_id_summary.csv", matches)
        write_global_summary_table(output_path.parent / "global_id_summary.csv", matches)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {output_path}")
    if args.write_tables:
        print(f"Saved mapping tables to: {output_path.parent}")


if __name__ == "__main__":
    main()
