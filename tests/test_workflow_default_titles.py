import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows" / "MiniMaxH3 - Long Video Digital Human.json"


def _load():
    return json.loads(WF.read_text(encoding="utf-8"))


def test_digital_human_workflow_has_clear_stage_titles():
    data = _load()
    titles = {node.get("title") for node in data["nodes"]}
    assert "AUTO TIMELINE — decoded audio duration → clip count" in titles
    assert "BASE — native Turbo 8-step / shared seed" in titles
    assert "DIGITAL HUMAN PROJECT — auto timeline → checkpoints → assemble" in titles


def test_static_numbered_clip_nodes_are_absent():
    data = _load()
    for node in data["nodes"]:
        title = node.get("title", "")
        if title.startswith("Clip ") and " - " in title:
            raise AssertionError(f"static numbered Clip node remains: {title}")


def test_live_final_path_is_absent():
    data = _load()
    types = {node["type"] for node in data["nodes"]}
    assert "MiniMaxH3StreamLiveDigitalHumanToVHS" not in types
    assert "MiniMaxH3LastActiveVHSPreviewBarrier" not in types
    assert "MiniMaxH3FinalizeVHSOutput" not in types
