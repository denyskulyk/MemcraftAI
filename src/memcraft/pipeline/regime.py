"""The regime rule: how a video is made, from the number of photos and the length.

One photo and a short video is multi-shot: the photo is the first frame and LTX invents the
rest. Two or three photos, or a long video, is keyframes: matching key pictures are drawn
first and LTX moves between them. The upload page and the worker both use this rule.
"""

from __future__ import annotations

from memcraft.models import JobLength, JobRegime
from memcraft.photos import MAX_PHOTOS


def regime_for(photo_count: int, length: JobLength) -> JobRegime:
    """The one place that decides how a video is made."""
    if not 1 <= photo_count <= MAX_PHOTOS:
        raise ValueError(f"photo_count must be 1 to {MAX_PHOTOS}")
    if photo_count == 1 and JobLength(length) is JobLength.SHORT:
        return JobRegime.MULTISHOT
    return JobRegime.KEYFRAMES
