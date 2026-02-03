from pathlib import Path


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "src").exists() and (parent / "assets").exists():
            return parent
    return cur.resolve()
