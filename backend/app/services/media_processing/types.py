"""
Shared data types for the media processing pipeline. Kept as plain
dataclasses (not Pydantic) since these never cross the HTTP boundary —
they're passed in-process from processor -> detectors.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MediaMetadata:
    width: int
    height: int
    duration_seconds: float | None = None  # None for images
    fps: float | None = None  # None for images
    frame_count: int | None = None  # total frames in the source video; None for images


@dataclass
class ExtractedFrame:
    """One frame, as read by OpenCV (BGR, uint8, HxWx3)."""

    index: int  # position in the original video; always 0 for images
    timestamp_seconds: float  # position in the original video; always 0.0 for images
    image: np.ndarray


@dataclass
class ProcessedMedia:
    metadata: MediaMetadata
    frames: list[ExtractedFrame]
