from typing import List, Optional
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor


class ReIDExtractor:
    def __init__(self, device: Optional[str] = None, model_name: str = "osnet_x0_25"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=self.device,
        )

    def __call__(self, rgb_crops: List[np.ndarray]) -> np.ndarray:
        if len(rgb_crops) == 0:
            return np.empty((0, 0), dtype=np.float32)

        feats = self.extractor(rgb_crops)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()

        return feats.astype(np.float32, copy=False)
