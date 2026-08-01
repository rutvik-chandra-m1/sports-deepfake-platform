"""
Reads dimension/timing metadata without extracting full frames — used
up front to decide sampling strategy and later surfaced in reports.
"""

import cv2

from app.models.analysis import MediaType
from app.services.media_processing.errors import MediaReadError
from app.services.media_processing.types import MediaMetadata


def read_metadata(file_path: str, media_type: MediaType) -> MediaMetadata:
    if media_type is MediaType.IMAGE:
        return _read_image_metadata(file_path)
    return _read_video_metadata(file_path)


def _read_image_metadata(file_path: str) -> MediaMetadata:
    image = cv2.imread(file_path)
    if image is None:
        raise MediaReadError(f"Could not read image file: {file_path}")

    height, width = image.shape[:2]
    return MediaMetadata(width=width, height=height)


def _read_video_metadata(file_path: str) -> MediaMetadata:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise MediaReadError(f"Could not open video file: {file_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    duration = (frame_count / fps) if fps and frame_count > 0 else None
    return MediaMetadata(
        width=width,
        height=height,
        duration_seconds=duration,
        fps=fps or None,
        frame_count=frame_count or None,
    )
