from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor


class ReIDExtractor:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "osnet_x1_0",
        model_weights_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "model_name": model_name,
            "device": self.device,
        }

        if model_weights_path:
            weights_path = str(Path(model_weights_path))
            try:
                self.extractor = FeatureExtractor(**kwargs, model_path=weights_path)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize torchreid FeatureExtractor with model_path. "
                    "Expected a clean state_dict checkpoint (saved via torch.save(model.state_dict(), ...)). "
                    f"Path: {weights_path}"
                ) from exc
        else:
            self.extractor = FeatureExtractor(**kwargs)

    def __call__(self, rgb_crops: List[np.ndarray]) -> np.ndarray:
        if len(rgb_crops) == 0:
            return np.empty((0, 0), dtype=np.float32)

        feats = self.extractor(rgb_crops)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()

        return feats.astype(np.float32, copy=False)
