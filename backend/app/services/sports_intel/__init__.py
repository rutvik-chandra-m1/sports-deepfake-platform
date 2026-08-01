"""Sports-specific intelligence (Milestone 12): jersey/team color
consistency, stadium/scene consistency, broadcast overlay tampering
checks, and crowd-texture duplication detection. All classical CV, no
sports-specific pretrained models (none exist off-the-shelf) -- see
docs/models.md for scope, limitations, and what was deliberately deferred
(athlete identity verification, match context verification)."""

from app.services.sports_intel.analysis import run_sports_intelligence
from app.services.sports_intel.broadcast_analysis import analyze_broadcast_overlay
from app.services.sports_intel.crowd_analysis import analyze_crowd_texture
from app.services.sports_intel.jersey_analysis import analyze_jersey_consistency
from app.services.sports_intel.scene_analysis import analyze_scene_consistency

__all__ = [
    "run_sports_intelligence",
    "analyze_broadcast_overlay",
    "analyze_crowd_texture",
    "analyze_jersey_consistency",
    "analyze_scene_consistency",
]
