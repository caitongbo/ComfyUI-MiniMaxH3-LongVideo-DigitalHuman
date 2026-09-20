import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_drift_control_standalone", ROOT / "h3_drift_control.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_schedule_matched_ratio_uses_next_sigma():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.0])
    assert MODULE.matched_noise_ratio(1.0, sigmas) == 0.75
    assert MODULE.matched_noise_ratio(0.75, sigmas) == 0.5 / 0.75
    assert MODULE.matched_noise_ratio(0.5, sigmas) == 0.0


def test_temporal_prefix_keeps_seam_side_clean():
    weights = MODULE.temporal_prefix_weights(12, 4)
    assert weights[:8] == (1.0,) * 8
    assert weights[-4:] == (0.75, 0.5, 0.25, 0.0)


def test_dynamic_mask_changes_video_only():
    video_shape = (1, 2, 14, 1, 1)
    audio_elements = 10
    packed = torch.zeros((1, 1, 28 + audio_elements))
    output, video = MODULE.apply_dynamic_prefix_mask(
        packed, video_shape, ratio=0.5, prefix_steps=12
    )
    assert tuple(video.shape) == (1, 1, 14, 1, 1)
    assert torch.equal(output[..., 28:], packed[..., 28:])
    assert float(video[0, 0, 0, 0, 0]) == 0.5
    assert float(video[0, 0, 11, 0, 0]) == 0.0

