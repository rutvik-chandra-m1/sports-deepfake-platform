"""
File validation helpers shared by the upload pipeline. Kept separate from
storage_service so the "what's a valid file" rules can be unit-tested (and
reused later for AI pipeline pre-checks) without touching disk.
"""

import re
from pathlib import Path

from app.core.config import Settings
from app.models.analysis import MediaType


class UnsupportedFileTypeError(Exception):
    """Raised when a file's extension isn't in the allowed image/video lists."""


class FileTooLargeError(Exception):
    """Raised when a file exceeds settings.max_upload_size_mb."""


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(filename: str) -> str:
    """Strip directory components and replace unsafe characters, so the
    result is safe to use as a path segment on disk."""
    name = Path(filename).name  # drop any ../ or directory components
    name = _UNSAFE_CHARS.sub("_", name)
    return name or "upload"


def resolve_media_type(filename: str, settings: Settings) -> MediaType:
    """Determine whether a filename is an allowed image or video, by extension."""
    ext = Path(filename).suffix.lower()

    if ext in settings.allowed_image_extensions_list:
        return MediaType.IMAGE
    if ext in settings.allowed_video_extensions_list:
        return MediaType.VIDEO

    allowed = ", ".join(settings.allowed_image_extensions_list + settings.allowed_video_extensions_list)
    raise UnsupportedFileTypeError(f"Unsupported file extension '{ext}'. Allowed: {allowed}")
