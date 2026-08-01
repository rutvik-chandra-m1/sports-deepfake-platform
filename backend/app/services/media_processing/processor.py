import logging

from app.models.analysis import MediaType
from app.services.media_processing.frame_extractor import DEFAULT_MAX_FRAMES, extract_frames
from app.services.media_processing.metadata import read_metadata
from app.services.media_processing.types import ProcessedMedia

logger = logging.getLogger(__name__)


def process_media(
    file_path: str, media_type: MediaType, max_frames: int = DEFAULT_MAX_FRAMES
) -> ProcessedMedia:
    """
    Entry point detectors (Milestones 7-8) will call: reads metadata, extracts
    sampled frames, and returns both bundled together. Raises MediaReadError
    if the file can't be opened/decoded.
    """
    metadata = read_metadata(file_path, media_type)
    frames = extract_frames(file_path, media_type, max_frames=max_frames)

    logger.info(
        "Processed %s (%s): %dx%d, %d frame(s) extracted",
        file_path,
        media_type.value,
        metadata.width,
        metadata.height,
        len(frames),
    )
    return ProcessedMedia(metadata=metadata, frames=frames)
