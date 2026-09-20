from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows"


def test_example_workflow_catalog_is_tight_and_current():
    names = sorted(p.name for p in WF.glob("*.json"))
    assert names == ["MiniMaxH3 - Long Video Digital Human.json"]


def test_no_legacy_workflow_families_are_distributed():
    names = [p.name for p in WF.glob("*.json")]
    assert not any(name.startswith(("NEW -", "OLD -")) for name in names)
