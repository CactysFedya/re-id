from pathlib import Path

from pipeline.config import load_pipeline_config
from pipeline.run_reid_file import run_file
from pipeline.run_reid_live import run_live
from pipeline.utils.paths import find_project_root
from pipeline.utils.sources import is_live_source


def main() -> None:
    project_root = find_project_root(Path(__file__))
    cfg = load_pipeline_config(project_root).reid
    runner = run_live if is_live_source(cfg.source.type) else run_file
    runner()


if __name__ == "__main__":
    main()
