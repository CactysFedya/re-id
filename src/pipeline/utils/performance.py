from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StageTimer:
    totals_s: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed_s: float, count: int = 0) -> None:
        self.totals_s[name] = self.totals_s.get(name, 0.0) + float(elapsed_s)
        if count:
            self.counts[name] = self.counts.get(name, 0) + int(count)

    def avg_ms(self, name: str, frames: int) -> float:
        if frames <= 0:
            return 0.0
        return self.totals_s.get(name, 0.0) / frames * 1000.0

    def total(self, name: str) -> int:
        return int(self.counts.get(name, 0))

    def summary(self, frames: int, exclude: set[str] | None = None) -> Dict[str, float]:
        excluded = exclude or set()
        return {
            f"avg_{name}_ms": round(self.avg_ms(name, frames), 4)
            for name in sorted(self.totals_s.keys())
            if name not in excluded
        }
