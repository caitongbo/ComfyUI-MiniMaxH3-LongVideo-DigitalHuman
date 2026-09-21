from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_only_checkpoint_streamer_is_public_and_experimental_paths_are_absent():
    nodes = (ROOT / "nodes.py").read_text(encoding="utf-8")
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    assert '"MiniMaxH3StreamLiveExtensionAVToVHS"' not in nodes
    assert '"MiniMaxH3StreamCheckpointDigitalHumanToVHS"' in nodes
    assert "class MiniMaxH3StreamLiveExtensionAVToVHS" not in stream
    assert "class MiniMaxH3StreamCheckpointDigitalHumanToVHS" in stream
    assert "MiniMaxH3StreamLiveDigitalHumanToVHS" not in nodes
    assert "MiniMaxH3LastActiveVHSPreviewBarrier" not in nodes
    assert "MiniMaxH3FinalizeVHSOutput" not in nodes
    assert "class MiniMaxH3StreamLiveDigitalHumanToVHS" not in stream
    assert "class MiniMaxH3LastActiveVHSPreviewBarrier" not in stream
    assert "class MiniMaxH3FinalizeVHSOutput" not in stream
    assert "MiniMaxH3LowRamPreviewToVHS" not in stream
    assert "VHS_BatchManager" not in stream
    assert "Meta Batch" not in stream
    # Superseded materialized final-movie assemblers must not remain registered
    # or available as alternative nodes: they recreated the large final RGB buffer.
    assert "MiniMaxH3AssembleLiveExtensionAV" not in nodes
    assert "MiniMaxH3AssembleLiveDigitalHuman" not in nodes
    assert "class MiniMaxH3AssembleLiveDigitalHuman" not in (ROOT / "h3_digital_human_audio_context.py").read_text(encoding="utf-8")


def test_streaming_runtime_uses_one_shot_checkpoint_frames():
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    assert "class _OneShotFrameSequence" in stream
    assert "materialized ``torch.Tensor``" in stream
    assert "def _checkpoint_frame_generator(" in stream
    assert 'tensors.get_tensor("video")' in stream
    assert "yield from _yield_segments_and_hold(" in stream
    assert "master_audio" in stream


def test_final_output_filename_allocation_stays_owned_by_vhs():
    """Guard against reintroducing fixed-path final-video overwrites."""
    stream = (ROOT / "h3_streaming_vhs.py").read_text(encoding="utf-8")
    wrapper = stream.split("def _run_vhs_h264(", 1)[1].split("\n\nclass MiniMaxH3StreamCheckpointDigitalHumanToVHS", 1)[0]

    # The H3 wrapper must hand the user prefix to stock VHS and let VHS choose
    # the numbered output path.  It must not become a custom file writer again.
    assert "vhs.combine_video(" in wrapper
    assert "filename_prefix=_expand_output_prefix(filename_prefix)" in wrapper
    for destructive in (
        "os.remove(", "os.unlink(", "os.replace(", "os.rename(",
        "Path.unlink(", "subprocess.", "ffmpeg",
    ):
        assert destructive not in wrapper


def test_digital_human_workflow_saves_checkpoint_output_with_stable_vhs_prefix():
    import json

    cases = [("MiniMaxH3 - Long Video Digital Human.json", "MiniMaxH3StreamCheckpointDigitalHumanToVHS",
              "video/minimax_h3_digital_human")]
    for filename, node_type, expected_prefix in cases:
        workflow = json.loads((ROOT / "example_workflows" / filename).read_text(encoding="utf-8"))
        final = next(node for node in workflow["nodes"] if node["type"] == node_type)
        values = final["widgets_values_named"]
        assert values["save_output"] is True
        assert values["filename_prefix"] == expected_prefix
        # Keep VHS's normal output bookkeeping enabled in the shipped examples.
        assert workflow.get("extra", {}).get("VHS_KeepIntermediate", True) is True
