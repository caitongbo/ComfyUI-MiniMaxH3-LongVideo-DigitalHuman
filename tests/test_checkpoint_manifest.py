import json
import os
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoint_manifest.py"


def _load_module(tmp_path, monkeypatch):
    folder_paths = types.SimpleNamespace(get_output_directory=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    spec = spec_from_file_location("_h3_checkpoint_manifest_test", SOURCE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def test_manifest_records_contiguous_lineage_and_detects_stale_descendants(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    first = Path(module.checkpoint_path(prefix, 1))
    second = Path(module.checkpoint_path(prefix, 2))
    _write(first, b"clip-one-v1")
    module.publish_checkpoint(prefix, 1, str(first), (1, 24, 2, 4, 4),
                              (1, 32, 2, 4))
    _write(second, b"clip-two-v1")
    module.publish_checkpoint(prefix, 2, str(second), (1, 24, 2, 4, 4),
                              (1, 32, 2, 4))
    assert module.validate_checkpoint_set(prefix, [str(first), str(second)])

    _write(first, b"clip-one-v2")
    module.publish_checkpoint(prefix, 1, str(first), (1, 24, 2, 4, 4),
                              (1, 32, 2, 4))
    with pytest.raises(ValueError, match="Clip 2 is stale"):
        module.validate_checkpoint_set(prefix, [str(first), str(second)])


def test_manifest_uses_unversioned_digital_human_format_and_rejects_old_format(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    first = Path(module.checkpoint_path(prefix, 1))
    _write(first, b"clip-one")
    module.publish_checkpoint(
        prefix, 1, str(first), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )
    path = Path(module.manifest_path(prefix))
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == "h3_digital_human_checkpoint_manifest"

    document["format"] = "unsupported_old_checkpoint_manifest"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported manifest format"):
        module.validate_checkpoint_set(prefix, [str(first)])


def test_manifest_dependency_signature_tracks_only_upstream_graph(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prompt = {
        "1": {"class_type": "Sampler", "inputs": {"seed": 10}},
        "2": {"class_type": "Save", "inputs": {
            "latent": ["1", 0], "checkpoint_prefix": "project/a", "clip_index": 1,
        }},
        "9": {"class_type": "Unrelated", "inputs": {"value": "a"}},
    }
    signature = module._dependency_signature(prompt, "2")
    prompt["9"]["inputs"]["value"] = "b"
    assert module._dependency_signature(prompt, "2") == signature
    prompt["2"]["inputs"]["checkpoint_prefix"] = "project/branch"
    assert module._dependency_signature(prompt, "2") == signature
    prompt["1"]["inputs"]["seed"] = 11
    assert module._dependency_signature(prompt, "2") != signature


def test_lazy_resume_requires_matching_upstream_signature(tmp_path, monkeypatch):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    path = Path(module.checkpoint_path(prefix, 1))
    prompt = {
        "1": {"class_type": "Sampler", "inputs": {"seed": 10}},
        "2": {"class_type": "ResumeGate", "inputs": {"latent": ["1", 0]}},
        "9": {"class_type": "Unrelated", "inputs": {"value": "a"}},
    }
    _write(path, b"checkpoint")
    module.publish_checkpoint(
        prefix, 1, str(path), (1, 24, 2, 4, 4), (1, 32, 2, 4),
        prompt=prompt, unique_id="2",
    )
    assert module.checkpoint_record_usable(
        prefix, 1, prompt, "2"
    ) == (True, "ready")
    prompt["9"]["inputs"]["value"] = "unrelated change"
    assert module.checkpoint_record_usable(prefix, 1, prompt, "2")[0] is True
    prompt["1"]["inputs"]["seed"] = 11
    usable, reason = module.checkpoint_record_usable(prefix, 1, prompt, "2")
    assert usable is False and reason == "upstream workflow changed"


def test_branch_prefix_change_does_not_invalidate_generation_signature(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    branch_prefix = "digital_human/checkpoints/branch"
    path = Path(module.checkpoint_path(prefix, 1))
    prompt = {
        "1": {"class_type": "Sampler", "inputs": {"seed": 10}},
        "2": {"class_type": "MiniMaxH3CheckpointResume", "inputs": {
            "latent": ["1", 0], "checkpoint_prefix": prefix,
            "clip_index": 1, "reuse_checkpoint": True,
        }},
    }
    _write(path, b"checkpoint")
    module.publish_checkpoint(
        prefix, 1, str(path), (1, 24, 2, 4, 4), (1, 32, 2, 4),
        prompt=prompt, unique_id="2",
    )
    module.branch_checkpoint_project(prefix, branch_prefix, 1)
    prompt["2"]["inputs"]["checkpoint_prefix"] = branch_prefix
    assert module.checkpoint_record_usable(
        branch_prefix, 1, prompt, "2"
    ) == (True, "ready")


def test_manifest_is_atomically_published_without_part_files(tmp_path, monkeypatch):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    path = Path(module.checkpoint_path(prefix, 1))
    _write(path, b"complete")
    target = module.publish_checkpoint(
        prefix, 1, str(path), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )
    document = json.loads(Path(target).read_text(encoding="utf-8"))
    assert document["format"] == module.FORMAT
    assert document["clips"]["1"]["sha256"] == module.file_sha256(path)
    assert not list(Path(target).parent.glob("*.part.*"))


def test_manifest_detects_same_size_content_replacement(tmp_path, monkeypatch):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    path = Path(module.checkpoint_path(prefix, 1))
    _write(path, b"original")
    module.publish_checkpoint(
        prefix, 1, str(path), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )
    original_stat = path.stat()
    _write(path, b"modified")
    assert path.stat().st_size == original_stat.st_size
    with pytest.raises(ValueError, match="content changed"):
        module.validate_checkpoint_set(prefix, [str(path)])


def test_changed_upstream_signature_stales_descendants_even_if_bytes_match(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    first = Path(module.checkpoint_path(prefix, 1))
    second = Path(module.checkpoint_path(prefix, 2))
    prompt = {
        "1": {"class_type": "Sampler", "inputs": {"seed": 10}},
        "2": {"class_type": "Save", "inputs": {"latent": ["1", 0]}},
    }
    _write(first, b"same-clip-one")
    module.publish_checkpoint(
        prefix, 1, str(first), (1, 24, 2, 4, 4), (1, 32, 2, 4),
        prompt=prompt, unique_id="2",
    )
    _write(second, b"clip-two")
    module.publish_checkpoint(
        prefix, 2, str(second), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )

    prompt["1"]["inputs"]["seed"] = 11
    module.publish_checkpoint(
        prefix, 1, str(first), (1, 24, 2, 4, 4), (1, 32, 2, 4),
        prompt=prompt, unique_id="2",
    )
    with pytest.raises(ValueError, match="Clip 2 is stale"):
        module.validate_checkpoint_set(prefix, [str(first), str(second)])


def test_overwrite_archives_immutable_take_and_can_branch_from_it(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    first = Path(module.checkpoint_path(prefix, 1))
    old_bytes = b"first-take"
    _write(first, old_bytes)
    module.publish_checkpoint(
        prefix, 1, str(first), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )
    old_sha = module.file_sha256(first)
    archive = module.archive_current_take(prefix, 1, str(first))
    replacement = first.with_suffix(".replacement")
    _write(replacement, b"second-take")
    os.replace(replacement, first)
    module.publish_checkpoint(
        prefix, 1, str(first), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )

    assert Path(archive).read_bytes() == old_bytes
    status = module.public_checkpoint_status(prefix)
    assert len(status["clips"][0]["takes"]) == 1
    assert status["clips"][0]["takes"][0]["sha256"] == old_sha
    assert status["storage"]["active_bytes"] == len(b"second-take")
    assert status["storage"]["take_bytes"] == len(old_bytes)

    branch = module.branch_checkpoint_project(
        prefix, "digital_human/checkpoints/old_take_branch", 1, old_sha
    )
    assert branch["clips"][0]["ready"] is True
    branched = Path(module.checkpoint_path(
        "digital_human/checkpoints/old_take_branch", 1
    ))
    assert branched.read_bytes() == old_bytes
    projects = module.list_checkpoint_projects()["projects"]
    assert {item["prefix"] for item in projects} == {
        prefix, "digital_human/checkpoints/old_take_branch"
    }


def test_uncommitted_post_replace_file_does_not_block_next_save(
    tmp_path, monkeypatch
):
    module = _load_module(tmp_path, monkeypatch)
    prefix = "digital_human/checkpoints/clip"
    path = Path(module.checkpoint_path(prefix, 1))
    _write(path, b"committed")
    module.publish_checkpoint(
        prefix, 1, str(path), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )
    # Simulate a kill after os.replace(checkpoint) and before manifest publish.
    _write(path, b"uncommitted-after-kill")
    assert module.archive_current_take(prefix, 1, str(path)) is None
    _write(path, b"recovered")
    module.publish_checkpoint(
        prefix, 1, str(path), (1, 24, 2, 4, 4), (1, 32, 2, 4)
    )
    assert module.validate_checkpoint_set(prefix, [str(path)])
