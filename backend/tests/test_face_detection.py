import numpy as np

from app.services.detection.face_detection import _cascade_path, detect_largest_face


def test_cascade_downloads_and_caches_on_first_use():
    # Exercises the real download path (raw.githubusercontent.com is
    # reachable in this environment) -- not mocked.
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    detect_largest_face(image)

    assert _cascade_path().exists()


def test_random_noise_has_no_detectable_face():
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = detect_largest_face(image)

    assert result is None
