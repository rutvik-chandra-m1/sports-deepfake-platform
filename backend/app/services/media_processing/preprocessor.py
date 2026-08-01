"""
Generic preprocessing shared by every detector. Deliberately minimal:
resize + BGR->RGB + scale to [0,1] float32. Model-specific normalization
(e.g. ImageNet mean/std, or a specific input size like 299x299 for
Xception vs 224x224 for a ViT) is applied by each detector individually in
Milestones 7-8, since different pretrained models expect different things.
"""

import cv2
import numpy as np


def preprocess_frame(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Returns an RGB float32 array in [0, 1], shape (H, W, 3) = (target_size[1], target_size[0], 3)."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0
