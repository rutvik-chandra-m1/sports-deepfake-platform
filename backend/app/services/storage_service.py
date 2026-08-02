"""
Storage service — everything about *where files live on disk* stays here,
behind a small function surface (`save_upload`). If cloud storage (S3, etc.)
replaces local disk later, only this file needs to change.
"""

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.security import (
    MAGIC_HEADER_BYTES,
    UnsupportedFileContentError,
    assert_content_matches_extension,
)
from app.models.analysis import MediaType
from app.utils.file_validation import (
    FileTooLargeError,
    resolve_media_type,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MB, streamed so large videos don't load fully into memory


def _subdirectory_for(media_type: MediaType, upload_dir: str) -> Path:
    base = Path(upload_dir)
    return base / ("images" if media_type is MediaType.IMAGE else "videos")


async def save_upload(file: UploadFile) -> tuple[str, MediaType, int]:
    """
    Validates and streams an uploaded file to disk under settings.upload_dir.

    Returns (stored_file_path, media_type, size_bytes).
    Raises UnsupportedFileTypeError / FileTooLargeError for invalid input —
    the API layer is responsible for turning those into HTTP error responses.
    """
    settings = get_settings()

    safe_name = sanitize_filename(file.filename or "upload")
    media_type = resolve_media_type(safe_name, settings)  # may raise UnsupportedFileTypeError

    destination_dir = _subdirectory_for(media_type, settings.upload_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Prefix with a UUID so concurrent uploads of the same filename never collide.
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    destination_path = destination_dir / unique_name

    size_bytes = 0
    header = b""
    try:
        with destination_path.open("wb") as buffer:
            while chunk := await file.read(_CHUNK_SIZE):
                # Validate the real file signature from the first chunk, before
                # committing the rest to disk (R9). Extension alone proves
                # nothing -- anything can be renamed .jpg, and the decoders
                # downstream are C++.
                if not header:
                    header = chunk[:MAGIC_HEADER_BYTES]
                    assert_content_matches_extension(header, Path(safe_name).suffix.lower())

                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_size_bytes:
                    raise FileTooLargeError(
                        f"File exceeds the {settings.max_upload_size_mb}MB upload limit."
                    )
                buffer.write(chunk)
    except (FileTooLargeError, UnsupportedFileContentError):
        # Never leave a rejected upload on disk.
        destination_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    logger.info("Saved upload -> %s (%s, %d bytes)", destination_path, media_type.value, size_bytes)
    return str(destination_path), media_type, size_bytes
