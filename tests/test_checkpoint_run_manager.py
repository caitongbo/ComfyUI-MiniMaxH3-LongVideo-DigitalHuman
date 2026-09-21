import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows" / "MiniMaxH3 - Long Video Digital Human.json"


def _load():
    return json.loads(WF.read_text(encoding="utf-8"))


def test_workflow_is_checkpoint_managed_by_one_reusable_gate():
    data = _load(); types = [node["type"] for node in data["nodes"]]
    assert types.count("MiniMaxH3CheckpointResume") == 1
    assert types.count("MiniMaxH3AudioTimelinePlan") == 1
    assert types.count("MiniMaxH3StreamCheckpointDigitalHumanToVHS") == 1
    assert "MiniMaxH3StreamLiveDigitalHumanToVHS" not in types
    gate = next(node for node in data["nodes"]
                if node["type"] == "MiniMaxH3CheckpointResume")
    assert gate["widgets_values_named"]["checkpoint_prefix"] == \
        "h3_digital_human/checkpoints/clip"
    assert gate["widgets_values_named"]["reuse_checkpoint"] is True
    assert gate["properties"]["h3_digital_human_role"] == \
        "reusable_checkpoint_gate"


def test_workflow_uses_one_direct_native_eight_step_sampler():
    data = _load(); nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}
    assert sum(node["type"] == "SamplerCustomAdvanced"
               for node in data["nodes"]) == 1
    seed = next(node for node in data["nodes"]
                if node.get("title") == "BASE SEED — fixed and shared across clips")
    assert seed["widgets_values_named"]["value"] == 514005817509111
    assert len(seed["outputs"][0]["links"]) == 1
    assert nodes[links[seed["outputs"][0]["links"][0]][3]]["type"] == "RandomNoise"
    scheduler = next(node for node in data["nodes"] if node["type"] == "BasicScheduler")
    assert scheduler["widgets_values"] == ["simple", 8, 1]


def test_manager_has_planning_retry_cleanup_preview_and_project_controls():
    source = (ROOT / "js" / "digital_human_checkpoint_manager.js").read_text(
        encoding="utf-8"
    )
    assert 'app.queuePrompt(0, 1, [String(target.id)])' in source
    assert "installNormalQueueRedirect" in source
    assert "normal Queue redirected to safe managed run" in source
    assert '"/api/queue"' in source
    assert '`${API_BASE}/cleanup`' in source
    assert "current.attempt < current.retries" in source
    assert "audio timeline planner" in source
    assert "Branch here" in source
    assert "Run / Resume / Retry" in source
    assert "Pause after current clip" in source
    assert "Assemble preview" in source
    assert "Final export" in source
    assert "Regenerate from here" in source
    assert "Preview clip" in source
    assert "expected one reusable checkpoint gate" in source
    routes = (ROOT / "checkpoint_routes.py").read_text(encoding="utf-8")
    assert 'routes.get(API_BASE)(status)' in routes
    assert 'routes.post(API_BASE + "/cleanup")(cleanup)' in routes


def test_checkpoint_loader_restores_native_nested_tensor_for_vae_decode():
    source = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert "comfy.nested_tensor.NestedTensor" in source
    assert '"samples": [data["video"], data["audio"]]' not in source
