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
        "MiniMaxH3DigitalHumanAVContext",
        "MiniMaxH3StreamCheckpointDigitalHumanToVHS",
        "MiniMaxH3AudioTimelinePlan",
    }


def test_workflow_has_exactly_one_reusable_generation_pipeline():
    data = load(); types = [node["type"] for node in data["nodes"]]
    for node_type in (
        "MiniMaxH3ReferenceToVideo", "MiniMaxH3DigitalHumanAVContext",
        "SamplerCustomAdvanced",
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
    reference = next(node for node in nodes.values() if node["type"] == "LoadImage")
    planner = next(node for node in nodes.values()
                   if node["type"] == "MiniMaxH3AudioTimelinePlan")
    assembler = next(node for node in nodes.values()
                     if node["type"] == "MiniMaxH3StreamCheckpointDigitalHumanToVHS")
    assert master["widgets_values"][0] == \
        "h3_digital_human_demo_narration.mp3"
    assert master["widgets_values_named"]["audio"] == master["widgets_values"][0]
    assert reference["widgets_values"][0] == \
        "h3_digital_human_presenter_reference.png"
    assert reference["widgets_values_named"]["image"] == \
        reference["widgets_values"][0]
    assert "/" not in master["widgets_values"][0]
    assert "/" not in reference["widgets_values"][0]
    assert links[_input(planner, "master_audio")["link"]][1] == master["id"]
    assert set(planner["widgets_values_named"]) == {
        "raw_clip_frames", "context_frames", "clip_index"
    }
    assert links[_input(assembler, "clip_count")["link"]][1] == planner["id"]
    assert links[_input(assembler, "master_audio")["link"]][1] == master["id"]


def test_one_managed_index_drives_context_gate_and_timeline():
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
        "MiniMaxH3AudioTimelinePlan",
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


def test_single_prompt_and_one_master_reference_feed_two_r2v_slots():
    data = load(); nodes = _nodes(data); links = _links(data)
    r2v = next(node for node in nodes.values()
               if node["type"] == "MiniMaxH3ReferenceToVideo")
    assert r2v["widgets_values_named"]["prompt"]
    prompt = r2v["widgets_values_named"]["prompt"]
    assert "<Picture 1>" in prompt
    assert "<Picture 2>" in prompt
    assert "<Audio 1>" in prompt
    assert "[reference generation + audio reuse]" in prompt
    assert "<Audio 1>:\n\nfully_copy -" in prompt
    assert "reused 1:1 as the target video's complete final audio track" in prompt
    assert "* add additional audio layers" in prompt
    assert "The same <Subject 1> identity must be preserved throughout all clips." in prompt
    assert "The camera setup remains identical throughout the clip." in prompt
    assert "Continue the exact previous frame state." in prompt
    assert "No visual reset." in prompt
    assert "No scene transition." in prompt
    section_names = [
        "subject_definitions:", "program_style:", "summary:",
        "retention_analysis:", "detailed_description:",
        "overall_soundscape:", "non_diegetic_music:",
    ]
    assert all(name in prompt for name in section_names)
    assert [prompt.index(name) for name in section_names] == sorted(
        prompt.index(name) for name in section_names
    )
    assert r2v["widgets_values"][0] == prompt
    refs = [node for node in nodes.values() if node["type"] == "LoadImage"]
    assert len(refs) == 1
    assert refs[0].get("mode", 0) == 0
    first = _input(r2v, "ref_images.ref_image_0")
    second = _input(r2v, "ref_images.ref_image_1")
    assert links[first["link"]][1] == refs[0]["id"]
    assert links[second["link"]][1] == refs[0]["id"]
    assert first["link"] != second["link"]
    for index in range(2, 5):
        assert _input(r2v, f"ref_images.ref_image_{index}").get("link") is None


def test_native_turbo_defaults_and_safe_disk_backed_final_output():
    data = load(); nodes = _nodes(data)
    scheduler = next(node for node in nodes.values() if node["type"] == "BasicScheduler")
    seed = next(node for node in nodes.values()
                if node.get("title") == "BASE SEED — fixed and shared across clips")
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
    assert "automatically calculates" in guide
    assert "independent checkpoint job" in guide
    groups = {group["title"] for group in data["groups"]}
    assert any(title.startswith("ONE REUSABLE CLIP PIPELINE") for title in groups)
    assert not any("controller managed" in title.lower() for title in groups)


def test_release_canvas_is_compact_non_overlapping_and_group_contained():
    data = load()
    nodes = data["nodes"]
    for index, first in enumerate(nodes):
        x1, y1 = first["pos"]
        w1, h1 = first["size"][:2]
        for second in nodes[index + 1:]:
            x2, y2 = second["pos"]
            w2, h2 = second["size"][:2]
            assert min(x1 + w1, x2 + w2) <= max(x1, x2) or \
                min(y1 + h1, y2 + h2) <= max(y1, y2)

    for node in nodes:
        x, y = node["pos"]
        width, height = node["size"][:2]
        containing = []
        for group in data["groups"]:
            gx, gy, gw, gh = group["bounding"]
            if gx <= x + width / 2 <= gx + gw and \
                    gy <= y + height / 2 <= gy + gh:
                containing.append(group)
        assert len(containing) == 1
        gx, gy, gw, gh = containing[0]["bounding"]
        assert gx <= x and gy <= y
        assert x + width <= gx + gw and y + height <= gy + gh

    left = min(node["pos"][0] for node in nodes)
    right = max(node["pos"][0] + node["size"][0] for node in nodes)
    top = min(node["pos"][1] for node in nodes)
    bottom = max(node["pos"][1] + node["size"][1] for node in nodes)
    assert right - left <= 6600
    assert bottom - top <= 2500
    assert data["extra"]["ds"] == {"scale": 0.72, "offset": [2600, 1650]}
    assert _nodes(data)[2601]["flags"]["collapsed"] is True


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
    inputs = ROOT / "example_inputs"
    assert (inputs / "presenter_reference.png").is_file()
    assert (inputs / "demo_narration.mp3").is_file()
    r2v = next(node for node in load()["nodes"]
               if node["type"] == "MiniMaxH3ReferenceToVideo")
    for item in r2v["inputs"]:
        if item["name"].startswith("ref_audios.ref_audio_"):
            assert item.get("link") is None
