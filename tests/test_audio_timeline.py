import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_timeline", ROOT / "h3_timeline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    ("seconds", "clips"),
    [(12.0, 1), (40.0, 3), (54.336, 4)],
)
def test_default_auto_timeline_covers_audio(seconds, clips):
    sample_rate = 32000
    plan = module.calculate_audio_timeline(
        round(seconds * sample_rate), sample_rate, 362, 39
    )
    assert plan["clip_count"] == clips
    assert plan["coverage_seconds"] >= seconds


def test_manual_override_preserves_automatic_diagnostic():
    plan = module.calculate_audio_timeline(
        40 * 32000, 32000, 362, 39, automatic=False, manual_clips=1
    )
    assert plan["clip_count"] == 1
    assert plan["automatic_clip_count"] == 3


def test_over_limit_audio_fails_before_generation():
    with pytest.raises(ValueError, match="at most 20"):
        module.calculate_audio_timeline(300 * 32000, 32000, 362, 39)
