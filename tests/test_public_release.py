import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def test_publication_metadata_and_community_files_are_present():
    required = {
        "README.md", "README_EN.md", "LICENSE", "NOTICE.md", "CHANGELOG.md",
        "CONTRIBUTING.md", "SECURITY.md", "RELEASING.md", "pyproject.toml", ".gitattributes",
        ".editorconfig", ".github/pull_request_template.md",
        ".github/CODEOWNERS", ".github/dependabot.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
    }
    assert not {name for name in required if not (ROOT / name).is_file()}
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "comfyui-minimaxh3-longvideo-digitalhuman"' in metadata
    assert 'version = "1.0.0"' in metadata
    assert 'PublisherId = "caitongbo"' in metadata
    assert 'web = "js"' in metadata
    package = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    version = next(
        node.value.value for node in package.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets)
    )
    assert f'version = "{version}"' in metadata


def test_published_workflows_contain_no_machine_absolute_paths_or_runtime_previews():
    for path in (ROOT / "example_workflows").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        strings = list(_strings(data))
        assert not [value for value in strings if value.startswith(("/home/", "/workspace/"))]
        assert not [value for value in strings if re.match(r"^[A-Za-z]:[\\/]", value)]
        assert "fullpath" not in json.dumps(data)
        published = json.dumps(data, ensure_ascii=False).lower()
        assert "huishan" not in published
        assert "talent group" not in published


def test_relative_markdown_links_resolve():
    failures = []
    for document in ROOT.rglob("*.md"):
        if ".git" in document.parts:
            continue
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            target = target.strip("<>")
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (document.parent / relative).exists():
                failures.append(f"{document.name}: {target}")
    assert not failures


def test_release_tree_does_not_include_models_checkpoints_or_secret_files():
    forbidden_names = {".env", "id_rsa", "id_ed25519"}
    forbidden_suffixes = {".ckpt", ".pt", ".pth", ".safetensors"}
    bad = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes:
            bad.append(path.relative_to(ROOT).as_posix())
    assert not bad


def test_release_tree_contains_no_retired_runtime_families_or_stubs():
    retired = {
        "existing_video_extension.py",
        "h3_auto_crop32.py",
        "h3_mask_precision.py",
        "h3_masked_bridge.py",
        "h3_v2v_fractional.py",
        "h3_workflow_utilities.py",
        "js/digital_human_video_preview.js",
    }
    assert not {name for name in retired if (ROOT / name).exists()}
    assert sorted(path.name for path in (ROOT / "js").glob("*.js")) == [
        "digital_human_checkpoint_manager.js",
        "digital_human_fixed_fps.js",
    ]
