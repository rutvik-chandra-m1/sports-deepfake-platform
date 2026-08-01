import cv2
import numpy as np
import pytest

from app.models.analysis import MediaType
from app.services.media_processing import MediaReadError, preprocess_frame, process_media
from app.services.media_processing.frame_extractor import extract_frames
from app.services.media_processing.metadata import read_metadata


def _make_image(tmp_path, width=120, height=80, name="frame.png"):
    path = tmp_path / name
    image = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


def _make_video(tmp_path, num_frames=30, fps=10, width=64, height=48, name="clip.mp4"):
    path = tmp_path / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # a distinct solid color per frame makes it easy to assert on identity later
        frame = np.full((height, width, 3), fill_value=(i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


# ----- Metadata -----


def test_read_metadata_for_image(tmp_path):
    path = _make_image(tmp_path, width=120, height=80)
    meta = read_metadata(path, MediaType.IMAGE)

    assert meta.width == 120
    assert meta.height == 80
    assert meta.duration_seconds is None
    assert meta.fps is None


def test_read_metadata_for_video(tmp_path):
    path = _make_video(tmp_path, num_frames=30, fps=10, width=64, height=48)
    meta = read_metadata(path, MediaType.VIDEO)

    assert meta.width == 64
    assert meta.height == 48
    assert meta.frame_count == 30
    assert meta.fps == pytest.approx(10, abs=1)
    assert meta.duration_seconds == pytest.approx(3.0, abs=0.5)


def test_read_metadata_raises_for_missing_image(tmp_path):
    with pytest.raises(MediaReadError):
        read_metadata(str(tmp_path / "does_not_exist.png"), MediaType.IMAGE)


def test_read_metadata_raises_for_missing_video(tmp_path):
    with pytest.raises(MediaReadError):
        read_metadata(str(tmp_path / "does_not_exist.mp4"), MediaType.VIDEO)


# ----- Frame extraction -----


def test_extract_frames_for_image_returns_single_frame(tmp_path):
    path = _make_image(tmp_path, width=100, height=50)
    frames = extract_frames(path, MediaType.IMAGE)

    assert len(frames) == 1
    assert frames[0].index == 0
    assert frames[0].timestamp_seconds == 0.0
    assert frames[0].image.shape == (50, 100, 3)


def test_extract_frames_for_video_respects_max_frames(tmp_path):
    path = _make_video(tmp_path, num_frames=50)
    frames = extract_frames(path, MediaType.VIDEO, max_frames=10)

    assert len(frames) == 10
    indices = [f.index for f in frames]
    assert indices == sorted(indices)  # ascending, evenly spaced
    assert indices[0] < 5  # near the start
    assert indices[-1] > 40  # near the end -> full-duration coverage, not just the opening


def test_extract_frames_for_video_returns_all_when_fewer_than_max(tmp_path):
    path = _make_video(tmp_path, num_frames=5)
    frames = extract_frames(path, MediaType.VIDEO, max_frames=32)

    assert len(frames) == 5


def test_extract_frames_raises_for_corrupt_video(tmp_path):
    bad_path = tmp_path / "corrupt.mp4"
    bad_path.write_bytes(b"this is not a real video file")

    with pytest.raises(MediaReadError):
        extract_frames(str(bad_path), MediaType.VIDEO)


# ----- Preprocessing -----


def test_preprocess_frame_resizes_and_normalizes():
    raw = np.random.randint(0, 255, size=(80, 120, 3), dtype=np.uint8)
    processed = preprocess_frame(raw, target_size=(224, 224))

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


# ----- End-to-end orchestration -----


def test_process_media_end_to_end_image(tmp_path):
    path = _make_image(tmp_path, width=100, height=50)
    result = process_media(path, MediaType.IMAGE)

    assert result.metadata.width == 100
    assert len(result.frames) == 1


def test_process_media_end_to_end_video(tmp_path):
    path = _make_video(tmp_path, num_frames=20)
    result = process_media(path, MediaType.VIDEO, max_frames=8)

    assert result.metadata.frame_count == 20
    assert len(result.frames) == 8
