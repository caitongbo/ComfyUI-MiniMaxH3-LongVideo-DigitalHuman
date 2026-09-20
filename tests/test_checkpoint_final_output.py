import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "h3_streaming_vhs.py"


def _load_stream_module():
    spec = spec_from_file_location("_h3_checkpoint_final_test_module", STREAM)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_paths_are_contiguous_and_confined_to_output(tmp_path, monkeypatch):
    module = _load_stream_module()
    folder_paths = types.SimpleNamespace(get_output_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    folder = tmp_path / "project" / "checkpoints"
    folder.mkdir(parents=True)
    for index in (1, 2):
        (folder / f"clip_{index:05d}.safetensors").write_bytes(b"checkpoint")

    paths = module._output_checkpoint_paths("project/checkpoints/clip", 2)
    assert paths == [
        str(folder / "clip_00001.safetensors"),
        str(folder / "clip_00002.safetensors"),
    ]
    with pytest.raises(ValueError, match="relative prefix"):
        module._output_checkpoint_paths("../outside/clip", 1)
    with pytest.raises(FileNotFoundError, match="Clip 3 checkpoint"):
        module._output_checkpoint_paths("project/checkpoints/clip", 3)


def test_decode_budget_refuses_likely_oom_before_decode(monkeypatch):
    module = _load_stream_module()
    monkeypatch.setattr(module, "_available_memory_gib", lambda: 8.0)
    # One 362-frame 1024x1024 float32 RGB clip is about 4.05 GiB before
    # decoder/encoder headroom, so an 8 GiB host is deliberately rejected.
    with pytest.raises(MemoryError, match="refusing final decode"):
        module._enforce_decode_memory_budget(
            [(1, 16, 92, 64, 64)], [362], [0], 39, 0.0
        )


def test_checkpoint_node_has_no_live_clip_inputs_and_is_an_output_node():
    module = _load_stream_module()
    cls = module.MiniMaxH3StreamCheckpointDigitalHumanToVHS
    schema = cls.INPUT_TYPES()
    assert cls.OUTPUT_NODE is True
    assert schema["required"]["checkpoint_prefix"][0] == "STRING"
    assert schema["required"]["clip_count"][0] == "INT"
    assert schema["required"]["video_overlap_frames"][1]["default"] == 0
    assert not any(name.startswith("clip_") for name in schema.get("optional", {}))
    assert "max_decode_ram_gb" in schema["required"]
    assert schema["required"]["decode_buffer"][1]["default"] == "disk-backed"


def test_supported_h3_vae_decodes_into_disk_mmap(tmp_path, monkeypatch):
    module = _load_stream_module()

    class FirstStage:
        comfy_has_chunked_io = True
        latents_mean = torch.zeros(24)

        @staticmethod
        def decode_output_shape(_shape):
            return (1, 3, 2, 4, 5)

        @staticmethod
        def decode(_samples, output_buffer=None):
            assert output_buffer is not None
            output_buffer.fill_(0.25)

    class VAE:
        first_stage_model = FirstStage()
        patcher = object()
        device = torch.device("cpu")
        vae_dtype = torch.float32
        disable_offload = False

        @staticmethod
        def memory_used_decode(_shape, _dtype):
            return 0

        @staticmethod
        def throw_exception_if_invalid():
            return None

    class DeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    management = types.SimpleNamespace(
        load_models_gpu=lambda *_a, **_k: None,
        cuda_device_context=lambda _device: DeviceContext(),
    )
    comfy = types.ModuleType("comfy")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)

    path = tmp_path / "decode.buffer"
    path.touch()
    images, mapped, mode = module._decode_checkpoint_video(
        VAE(), torch.zeros((1, 24, 2, 1, 1)), "disk-backed", str(path)
    )
    assert mode == "disk-backed mmap"
    assert tuple(images.shape) == (2, 4, 5, 3)
    assert mapped is not None
    assert float(images.mean()) == pytest.approx(0.25)


def test_atomic_checkpoint_publication_is_present():
    nodes = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert ".part.%s.%s" in nodes
    assert 'open(temporary, "rb+")' in nodes
    assert "os.fsync(saved.fileno())" in nodes
    assert "os.replace(temporary, path)" in nodes
    assert "MiniMaxH3StreamCheckpointDigitalHumanToVHS" in nodes
    assert "class MiniMaxH3CheckpointResume" in nodes
    assert '"lazy": True' in nodes
    assert "archive_current_take" in nodes
