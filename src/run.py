from pathlib import Path

from pipeline.cli import main
from pipeline.utils.paths import find_project_root


def _default_repo_config() -> Path:
    project_root = find_project_root(Path(__file__))
    return project_root / "configs" / "pipeline.toml"


if __name__ == "__main__":
    main(["--config", str(_default_repo_config())])
