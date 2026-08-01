"""
Frame extraction.

Images yield a single frame. Videos are sampled *evenly across the whole
duration* (not just the first N frames) up to `max_frames`, so downstream
temporal-consistency analysis (Milestone 11) sees representative coverage
rather than just the opening seconds.
"""

import cv2

from app.models.analysis import MediaType
from app.services.media_processing.errors import MediaReadError
from app.services.media_processing.types import ExtractedFrame

DEFAULT_MAX_FRAMES = 32


def extract_frames(
    file_path: str, media_type: MediaType, max_frames: int = DEFAULT_MAX_FRAMES
) -> list[ExtractedFrame]:
    if media_type is MediaType.IMAGE:
        return _extract_image_frame(file_path)
    return _extract_video_frames(file_path, max_frames)


def _extract_image_frame(file_path: str) -> list[ExtractedFrame]:
    image = cv2.imread(file_path)
    if image is None:
        raise MediaReadError(f"Could not read image file: {file_path}")
    return [ExtractedFrame(index=0, timestamp_seconds=0.0, image=image)]


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count >= total:
        return list(range(total))
    step = total / count
    # dedupe in case of rounding collisions when count is close to total
    return sorted({int(i * step) for i in range(count)})


def _extract_video_frames(file_path: str, max_frames: int) -> list[ExtractedFrame]:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise MediaReadError(f"Could not open video file: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        if reported_total > 0:
            wanted_indices = set(evenly_spaced_indices(reported_total, max_frames))
            extracted: list[ExtractedFrame] = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted_indices:
                    timestamp = (idx / fps) if fps else 0.0
                    extracted.append(ExtractedFrame(index=idx, timestamp_seconds=timestamp, image=frame))
                idx += 1
        else:
            # Some containers/codecs don't report frame count reliably —
            # fall back to reading sequentially and subsampling afterward.
            all_frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                all_frames.append(frame)

            wanted_indices = evenly_spaced_indices(len(all_frames), max_frames)
            extracted = [
                ExtractedFrame(
                    index=i,
                    timestamp_seconds=(i / fps) if fps else 0.0,
                    image=all_frames[i],
                )
                for i in wanted_indices
            ]
    finally:
        cap.release()

    if not extracted:
        raise MediaReadError(f"No frames could be read from: {file_path}")

    return extracted
