from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MOT17 sequence directory with img1/*.jpg frames to a video file."
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Path to a MOT17 sequence directory, for example data/MOT17/train/MOT17-02-SDP.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to data/MOT17/videos/<sequence-name>.avi.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for a shorter test video.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="First 1-based MOT17 frame to include. Defaults to 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output video if it already exists.",
    )
    return parser.parse_args()


def load_seqinfo(sequence_dir: Path) -> dict[str, str]:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_path}")

    config = configparser.ConfigParser()
    config.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in config:
        raise ValueError(f"Invalid seqinfo.ini without [Sequence]: {seqinfo_path}")

    return dict(config["Sequence"])


def default_output_path(sequence_dir: Path, sequence_name: str) -> Path:
    mot17_root = sequence_dir.parent.parent
    return mot17_root / "videos" / f"{sequence_name}.avi"


def convert_sequence(
    sequence_dir: Path,
    output_path: Path,
    max_frames: int | None,
    overwrite: bool,
    start_frame: int = 1,
) -> None:
    sequence_dir = sequence_dir.resolve()
    info = load_seqinfo(sequence_dir)

    sequence_name = info.get("name", sequence_dir.name)
    frames_dir = sequence_dir / info.get("imdir", "img1")
    fps = float(info.get("framerate", 30))
    width = int(info["imwidth"])
    height = int(info["imheight"])
    frame_count = int(info["seqlength"])
    image_ext = info.get("imext", ".jpg")

    start_frame = max(1, int(start_frame))
    if start_frame > frame_count:
        raise ValueError(f"start_frame={start_frame} exceeds sequence length {frame_count}")

    end_frame = frame_count
    if max_frames is not None:
        end_frame = min(frame_count, start_frame + max(0, max_frames) - 1)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    try:
        written = 0
        for frame_idx in range(start_frame, end_frame + 1):
            frame_path = frames_dir / f"{frame_idx:06d}{image_ext}"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"Converted {written} frames from {sequence_name} to {output_path}")
    print(f"Frame range: {start_frame}-{end_frame}")
    print(f"Video properties: {width}x{height}, {fps:g} FPS")


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir
    info = load_seqinfo(sequence_dir)
    output_path = args.output or default_output_path(sequence_dir, info.get("name", sequence_dir.name))
    convert_sequence(sequence_dir, output_path, args.max_frames, args.overwrite, args.start_frame)


if __name__ == "__main__":
    main()
