from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.cli import _infer_source_type
from pipeline.config import load_pipeline_config, merge_config_overrides
from pipeline.utils.paths import find_project_root, resolve_path


class PipelineHelperTests(unittest.TestCase):
    def test_infer_source_type_from_uri(self) -> None:
        self.assertEqual(_infer_source_type("rtsp://127.0.0.1/stream", None), "rtsp")
        self.assertEqual(_infer_source_type("https://example.com/video", None), "https")
        self.assertEqual(_infer_source_type("http://example.com/video", None), "http")
        self.assertEqual(_infer_source_type("assets/demo.mp4", None), "file")

    def test_infer_source_type_from_device_index(self) -> None:
        self.assertEqual(_infer_source_type(None, 0), "camera")

    def test_resolve_path_returns_absolute_for_relative_input(self) -> None:
        resolved = resolve_path(PROJECT_ROOT, "configs/pipeline.toml")
        self.assertEqual(resolved, (PROJECT_ROOT / "configs" / "pipeline.toml").resolve())

    def test_resolve_path_returns_none_for_blank_input(self) -> None:
        self.assertIsNone(resolve_path(PROJECT_ROOT, ""))
        self.assertIsNone(resolve_path(PROJECT_ROOT, None))

    def test_find_project_root_from_src_file(self) -> None:
        root = find_project_root(PROJECT_ROOT / "src" / "run.py")
        self.assertEqual(root, PROJECT_ROOT.resolve())

    def test_merge_config_overrides_deep_merges_sections(self) -> None:
        merged = merge_config_overrides(
            {"reid": {"source": {"type": "rtsp"}}},
            {"reid": {"source": {"uri": "rtsp://127.0.0.1/live"}}},
        )
        self.assertEqual(merged["reid"]["source"]["type"], "rtsp")
        self.assertEqual(merged["reid"]["source"]["uri"], "rtsp://127.0.0.1/live")

    def test_invalid_detector_conf_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_pipeline_config(overrides={"reid": {"detector": {"conf": 2.0}}})


if __name__ == "__main__":
    unittest.main()
