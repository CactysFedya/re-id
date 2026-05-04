from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / ".tmp"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.config import dump_default_config, load_pipeline_config, resolve_runtime_base_dir, write_default_config


def make_case_dir(name: str) -> Path:
    case_dir = TEST_TMP_ROOT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


class PipelineConfigTests(unittest.TestCase):
    def test_loads_builtin_defaults_without_file(self) -> None:
        cfg = load_pipeline_config()
        self.assertEqual(cfg.reid.source.type, "file")
        self.assertEqual(cfg.reid.outputs_root, "outputs")
        self.assertEqual(cfg.reid.detector.model_name, "yolo26n.pt")

    def test_partial_toml_overrides_defaults(self) -> None:
        case_dir = make_case_dir("partial_override")
        config_path = case_dir / "pipeline.toml"
        config_path.write_text(
            """
[reid.source]
type = "rtsp"
uri = "rtsp://127.0.0.1:8554/stream"

[reid.detector]
conf = 0.12
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config(config_path)
        self.assertEqual(cfg.reid.source.type, "rtsp")
        self.assertEqual(cfg.reid.source.uri, "rtsp://127.0.0.1:8554/stream")
        self.assertAlmostEqual(cfg.reid.detector.conf, 0.12)
        self.assertEqual(cfg.reid.extractor.model_name, "osnet_x1_0")

    def test_runtime_base_dir_uses_config_parent(self) -> None:
        case_dir = make_case_dir("base_dir")
        nested = case_dir / "configs"
        nested.mkdir(parents=True, exist_ok=True)
        config_path = nested / "pipeline.toml"
        config_path.write_text("[reid]\nrun_prefix = \"demo\"\n", encoding="utf-8")

        base_dir = resolve_runtime_base_dir(config_path)
        self.assertEqual(base_dir, nested.resolve())

    def test_can_write_template_file(self) -> None:
        case_dir = make_case_dir("write_template")
        output_path = case_dir / "pipeline.template.toml"
        saved_path = write_default_config(output_path, overwrite=True)
        self.assertEqual(saved_path, output_path.resolve())
        self.assertEqual(output_path.read_text(encoding="utf-8"), dump_default_config())


if __name__ == "__main__":
    unittest.main()
