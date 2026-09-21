"""Behavioral regressions for the direct checkpoint frame streamer."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_stream_module():
    package_name = "_h3_stream_equivalence_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.h3_streaming_vhs", ROOT / "h3_streaming_vhs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _full_buffer_reference(clips, contexts, overlap):
    """Reference the expected linear seam math in-test."""
    out = clips[0].clone()
    for i in range(1, len(clips)):
        decoded = clips[i]
        ctx = int(contexts[i])
        ov = min(int(overlap), ctx, int(out.shape[0]))
        if ov > 0:
            dst = decoded[ctx - ov : ctx]
            alpha = torch.linspace(0.0, 1.0, ov + 2, dtype=torch.float32)[1:-1].view(
                -1, 1, 1, 1
            )
            out[-ov:].mul_(1.0 - alpha).add_(dst * alpha)
        out = torch.cat((out, decoded[ctx:]), dim=0)
    return out


def test_streamed_checkpoint_frames_match_full_buffer_reference(tmp_path):
    stream = _load_stream_module()
    stream._release_decode_memory = lambda: None

    # Small deterministic RGB clips with context larger than visual overlap.
    clips = [
        torch.arange(8 * 2 * 3 * 3, dtype=torch.float32).view(8, 2, 3, 3) / 1000.0,
        torch.arange(1000, 1000 + 9 * 2 * 3 * 3, dtype=torch.float32).view(9, 2, 3, 3) / 1000.0,
        torch.arange(2000, 2000 + 7 * 2 * 3 * 3, dtype=torch.float32).view(7, 2, 3, 3) / 1000.0,
    ]
    raw_frames = [len(x) for x in clips]
    contexts = [0, 3, 3]
    overlap = 2

    expected = _full_buffer_reference(clips, contexts, overlap)

    paths = [f"clip-{index}" for index in range(len(clips))]
    by_path = dict(zip(paths, clips))

    class SafeOpen:
        def __init__(self, path, **_kwargs):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_tensor(self, name):
            assert name == "video"
            return by_path[self.path]

    stream._safe_open = SafeOpen
    stream._decode_checkpoint_video = (
        lambda _vae, latent, _mode, _buffer: (latent.clone(), None, "memory")
    )
    buffers = iter(tmp_path / f"decode-{index}.buffer" for index in range(len(clips)))

    def next_buffer():
        path = next(buffers)
        path.touch()
        return str(path)

    stream._new_decode_buffer_path = next_buffer
    actual = torch.stack(
        list(
            stream._checkpoint_frame_generator(
                None,
                paths,
                raw_frames,
                contexts,
                overlap,
                "memory",
                "test_stream",
            )
        )
    )
    assert torch.equal(actual, expected)


def test_one_shot_vhs_sequence_primes_clip_zero_only_once():
    stream = _load_stream_module()
    calls = {"factory": 0, "yielded": 0}

    def factory():
        calls["factory"] += 1
        for i in range(4):
            calls["yielded"] += 1
            yield torch.full((2, 3, 3), float(i))

    seq = stream._OneShotFrameSequence(4, factory)
    assert len(seq) == 4
    assert torch.equal(seq[0], torch.zeros((2, 3, 3)))
    frames = list(seq)
    assert [float(x[0, 0, 0]) for x in frames] == [0.0, 1.0, 2.0, 3.0]
    assert calls == {"factory": 1, "yielded": 4}

    try:
        list(seq)
    except RuntimeError as exc:
        assert "one-shot" in str(exc)
    else:
        raise AssertionError("the internal VHS frame sequence must not be iterated twice")
