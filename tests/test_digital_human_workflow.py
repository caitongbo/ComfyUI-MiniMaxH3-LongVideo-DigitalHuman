import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows" / "MiniMaxH3 - Long Video Digital Human.json"


def load():
    return json.loads(WF.read_text(encoding="utf-8"))


def _nodes(data):
    return {node["id"]: node for node in data["nodes"]}


def _links(data):
    return {link[0]: link for link in data["links"]}


def _input(node, name):
    return next(item for item in node.get("inputs", []) if item["name"] == name)


def test_public_registry_exposes_only_current_single_pipeline_nodes():
    tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                for target in node.targets)
    )
    assert {item.value for item in assignment.value.keys} == {
        "MiniMaxH3AudioVAECompatibility", "MiniMaxH3CheckpointResume",
        "MiniMaxH3DigitalHumanAVContext", "MiniMaxH3DriftControlModel",
        "MiniMaxH3StreamCheckpointDigitalHumanToVHS",
        "MiniMaxH3AudioTimelinePlan",
    }


def test_workflow_has_exactly_one_reusable_generation_pipeline():
    data = load(); types = [node["type"] for node in data["nodes"]]
    for node_type in (
        "MiniMaxH3ReferenceToVideo", "MiniMaxH3DigitalHumanAVContext",
        "MiniMaxH3DriftControlModel", "SamplerCustomAdvanced",
        "MiniMaxH3CheckpointResume", "VAEDecode", "VHS_VideoCombine",
        "MiniMaxH3AudioTimelinePlan",
        "MiniMaxH3StreamCheckpointDigitalHumanToVHS",
    ):
        assert types.count(node_type) == 1
    assert "MiniMaxH3DigitalHumanController" not in types
    assert len(data["nodes"]) < 45


def test_auto_planner_uses_master_audio_and_drives_final_clip_count():
    data = load(); nodes = _nodes(data); links = _links(data)
    master = next(node for node in nodes.values() if node["type"] == "LoadAudio")
    planner = next(node for node in nodes.values()
                   if node["type"] == "MiniMaxH3AudioTimelinePlan")
    assembler = next(node for node in nodes.values()
                     if node["type"] == "MiniMaxH3StreamCheckpointDigitalHumanToVHS")
    assert master["widgets_values"][0] == "h3_digital_human/demo_narration.mp3"
    assert links[_input(planner, "master_audio")["link"]][1] == master["id"]
    assert planner["widgets_values_named"]["clip_mode"] == "Auto from Audio"
    assert planner["widgets_values_named"]["manual_clips"] == 4
    assert links[_input(assembler, "clip_count")["link"]][1] == planner["id"]
    assert links[_input(assembler, "master_audio")["link"]][1] == master["id"]


def test_one_managed_index_drives_context_gate_and_drift_control():
    data = load(); nodes = _nodes(data); links = _links(data)
    index = next(node for node in nodes.values()
                 if node.get("properties", {}).get("h3_digital_human_role")
                 == "h3_digital_human_current_clip")
    target_types = {
        nodes[links[link_id][3]]["type"]
        for link_id in index["outputs"][0]["links"]
    }
    assert target_types == {
        "MiniMaxH3DigitalHumanAVContext", "MiniMaxH3CheckpointResume",
        "MiniMaxH3DriftControlModel", "MiniMaxH3AudioTimelinePlan",
    }
    assert index["widgets_values_named"]["value"] == 1


def test_reusable_context_owns_checkpoint_continuation_and_exact_audio():
    data = load(); nodes = _nodes(data); links = _links(data)
    context = next(node for node in nodes.values()
                   if node["type"] == "MiniMaxH3DigitalHumanAVContext")
    assert context["widgets_values_named"]["checkpoint_prefix"] == \
        "h3_digital_human/checkpoints/clip"
    assert context["widgets_values_named"]["context_length"] == 39
    assert not {"source_latent", "source_frames", "clip_start_seconds"} & {
        item["name"] for item in context["inputs"]
    }
    gate = next(node for node in nodes.values()
                if node["type"] == "MiniMaxH3CheckpointResume")
    sampler = nodes[links[_input(gate, "latent")["link"]][1]]
    assert sampler["type"] == "SamplerCustomAdvanced"
    prepared = nodes[links[_input(sampler, "latent_image")["link"]][1]]
    assert prepared["id"] == context["id"]


def test_single_prompt_and_four_global_references_feed_reusable_r2v():
    data = load(); nodes = _nodes(data); links = _links(data)
    r2v = next(node for node in nodes.values()
               if node["type"] == "MiniMaxH3ReferenceToVideo")
    assert r2v["widgets_values_named"]["prompt"]
    refs = [node for node in nodes.values() if node["type"] == "LoadImage"]
    assert len(refs) == 4
    assert sum(node.get("mode", 0) == 0 for node in refs) == 2
    assert sum(node.get("mode", 0) == 4 for node in refs) == 2
    sockets = [_input(r2v, f"ref_images.ref_image_{index}") for index in range(4)]
    assert {links[item["link"]][1] for item in sockets} == {
        node["id"] for node in refs
    }


def test_native_turbo_defaults_and_safe_disk_backed_final_output():
    data = load(); nodes = _nodes(data)
    scheduler = next(node for node in nodes.values() if node["type"] == "BasicScheduler")
    seed = next(node for node in nodes.values()
                if node.get("title") == "BASE — native Turbo 8-step / shared seed")
    assembler = next(node for node in nodes.values()
                     if node["type"] == "MiniMaxH3StreamCheckpointDigitalHumanToVHS")
    assert scheduler["widgets_values"] == ["simple", 8, 1]
    assert seed["widgets_values_named"]["value"] == 514005817509111
    assert assembler["widgets_values_named"]["decode_buffer"] == "disk-backed"
    assert assembler["widgets_values_named"]["video_overlap_frames"] == 0
    assert assembler["widgets_values_named"]["trim_to_audio"] is True
    assert assembler["widgets_values_named"]["save_output"] is True


def test_current_clip_preview_is_single_reusable_output():
    data = load(); nodes = _nodes(data); links = _links(data)
    preview = next(node for node in data["nodes"]
                   if node["type"] == "VHS_VideoCombine")
    assert preview["properties"]["h3_digital_human_role"] == "current_clip_preview"
    assert preview["widgets_values_named"]["frame_rate"] == 24
    assert preview["widgets_values_named"]["save_output"] is False
    assert nodes[links[_input(preview, "audio")["link"]][1]]["type"] == \
        "MiniMaxH3AudioTimelinePlan"


def test_workflow_links_are_internally_consistent():
    data = load(); nodes = _nodes(data); links = _links(data)
    for node in nodes.values():
        for slot, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if link_id is not None:
                assert link_id in links
                assert links[link_id][3:5] == [node["id"], slot]
        for slot, item in enumerate(node.get("outputs", [])):
            for link_id in item.get("links") or []:
                assert links[link_id][1:3] == [node["id"], slot]


def test_frontend_queues_plan_then_reuses_one_gate_per_clip():
    source = (ROOT / "js" / "digital_human_checkpoint_manager.js").read_text(
        encoding="utf-8"
    )
    assert 'phase: "plan"' in source
    assert 'nodeType(node) === PLANNER' in source
    assert "expected one reusable checkpoint gate" in source
    assert "indexWidget.value = current.index" in source
    assert 'app.queuePrompt(0, 1, [String(target.id)])' in source
    assert 'api.addEventListener("executed"' in source
    assert 'api.addEventListener("execution_success"' in source
    assert '`${API_BASE}/cleanup`' in source
    assert "Preview clip" in source


def test_workflow_guidance_and_layout_describe_one_pipeline():
    data = load(); nodes = _nodes(data)
    guide = nodes[900]["widgets_values_named"]["text"]
    assert "one prompt" in guide.lower()
    assert "Auto from Audio" in guide
    assert "independent checkpoint job" in guide
    groups = {group["title"] for group in data["groups"]}
    assert "ONE REUSABLE CLIP PIPELINE" in groups
    assert not any("controller managed" in title.lower() for title in groups)


def test_single_pipeline_keeps_both_h3_clocks_locked_to_24_fps():
    source = (ROOT / "js" / "digital_human_fixed_fps.js").read_text(
        encoding="utf-8"
    )
    assert 'new Set(["fps", "frame_rate", "source_fps"' in source
    assert "widget.options.min = 24" in source
    assert "widget.options.max = 24" in source
    data = load()
    clock_names = {"frame_rate", "source_fps"}
    clocked = [
        node for node in data["nodes"]
        if clock_names & set(node.get("widgets_values_named", {}))
    ]
    assert len(clocked) == 2
    assert all(node.get("properties", {}).get("h3_fixed_fps") is True
               for node in clocked)


def test_bundled_inputs_exist_and_ref_audio_sockets_are_empty():
    inputs = ROOT / "example_inputs" / "h3_digital_human"
    assert (inputs / "presenter_reference.png").is_file()
    assert (inputs / "demo_narration.mp3").is_file()
    r2v = next(node for node in load()["nodes"]
               if node["type"] == "MiniMaxH3ReferenceToVideo")
    for item in r2v["inputs"]:
        if item["name"].startswith("ref_audios.ref_audio_"):
            assert item.get("link") is None
