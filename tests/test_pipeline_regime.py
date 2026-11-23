import pytest

from memcraft.models import JobLength, JobRegime
from memcraft.pipeline.regime import regime_for
from memcraft.web import upload

L, R = JobLength, JobRegime


@pytest.mark.parametrize(
    ("photos", "length", "regime"),
    [
        (1, L.SHORT, R.MULTISHOT),
        (1, L.LONG, R.KEYFRAMES),
        (2, L.SHORT, R.KEYFRAMES),
        (2, L.LONG, R.KEYFRAMES),
        (3, L.SHORT, R.KEYFRAMES),
        (3, L.LONG, R.KEYFRAMES),
    ],
)
def test_the_six_cases(photos, length, regime):
    assert regime_for(photos, length) is regime
    assert regime_for(photos, length.value) is regime


@pytest.mark.parametrize("photos", [0, 4])
def test_photo_count_out_of_range(photos):
    with pytest.raises(ValueError, match="photo_count"):
        regime_for(photos, L.SHORT)


def test_the_upload_uses_the_same_rule():
    assert upload.regime_for is regime_for
