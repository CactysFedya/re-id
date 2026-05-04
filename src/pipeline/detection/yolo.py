from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    xyxy: Tuple[int, int, int, int]
    conf: float
    cls: int


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        weights_path: Optional[Union[str, Path]] = None,
        conf: float = 0.25,
        classes=None,
    ):
        self.conf = float(conf)
        self.classes = classes if classes is not None else [0]

        weights_spec = model_name
        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                weights_spec = str(p)

        self.model = YOLO(weights_spec)

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            frame_bgr,
            verbose=False,
            conf=self.conf,
            classes=self.classes,
        )[0]

        dets: List[Detection] = []
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            dets.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    conf=float(b.conf.item()),
                    cls=int(b.cls.item()),
                )
            )
        return dets
