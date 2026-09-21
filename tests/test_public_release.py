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
        "CONTRIBUTING.md", "SECURITY.md", "docs/README.md",
        "docs/RELEASING.md", "docs/TECHNICAL_ARCHITECTURE.md",
        "docs/WORKFLOW_PREREQUISITES.md", "example_outputs/README.md",
        "pyproject.toml", ".gitattributes",
        ".editorconfig", ".comfyignore", ".github/pull_request_template.md",
        ".github/CODEOWNERS", ".github/dependabot.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
    }
    assert not {name for name in required if not (ROOT / name).is_file()}
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "minimaxh3-longvideo-digitalhuman"' in metadata
    assert 'version = "1.0.0"' in metadata
    assert 'PublisherId = "caitongbo"' in metadata
    assert 'requires-python = ">=3.10"' in metadata
    assert 'keywords = ["comfyui", "minimax-h3"' in metadata
    comfyignore = (ROOT / ".comfyignore").read_text(encoding="utf-8")
    for development_path in (".github/", "tests/", "requirements-test.txt", "pytest.ini"):
        assert development_path in comfyignore
    assert "example_outputs/*.mp4" in comfyignore
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
    allowed_showcase = {"example_outputs/digital_human_demo.mp4"}
    forbidden_names = {".env", "id_rsa", "id_ed25519"}
    forbidden_suffixes = {
        ".ckpt", ".pt", ".pth", ".safetensors",
        ".log", ".zip", ".7z", ".mp4", ".mkv", ".mov", ".webm", ".avi",
    }
    bad = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if (path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes) \
                and relative not in allowed_showcase:
            bad.append(relative)
    assert not bad
    demo = ROOT / "example_outputs" / "digital_human_demo.mp4"
    if demo.exists():
        assert demo.stat().st_size <= 50 * 1024 * 1024
        with demo.open("rb") as handle:
            assert handle.read(12)[4:8] == b"ftyp"


def test_release_tree_contains_only_supported_root_runtime_modules():
    expected = {
        "__init__.py", "checkpoint_manifest.py", "checkpoint_routes.py",
        "h3_audio_grid.py", "h3_audio_vae_compat.py", "h3_compat.py",
        "h3_digital_human_audio_context.py", "h3_mask_compat.py",
        "h3_mask_payload_compat.py", "h3_mask_velocity.py",
        "h3_media_utils.py", "h3_streaming_vhs.py", "h3_timeline.py",
        "h3_timing.py", "nodes.py",
    }
    assert {path.name for path in ROOT.glob("*.py")} == expected
    assert sorted(path.name for path in (ROOT / "js").glob("*.js")) == [
        "digital_human_checkpoint_manager.js",
        "digital_human_fixed_fps.js",
    ]


def test_public_docs_describe_only_the_current_release():
    documents = [
        ROOT / "README.md", ROOT / "README_EN.md",
        ROOT / "docs" / "WORKFLOW_PREREQUISITES.md",
        ROOT / "example_workflows" / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    assert "Auto from Audio" not in text
    assert "manual_clips" not in text
    assert "timeline mode" not in text.lower()
    assert "Upgrading from" not in text


def test_initial_release_metadata_has_no_internal_schema_labels():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [1.0.0]" in changelog
    assert "## [2." not in changelog
    workflow = json.loads(
        (ROOT / "example_workflows" / "MiniMaxH3 - Long Video Digital Human.json")
        .read_text(encoding="utf-8")
    )
    assert workflow["revision"] == 1
    properties = [node.get("properties", {}) for node in workflow["nodes"]]
    assert not [value for props in properties for value in props.values()
                if isinstance(value, str) and re.search(r"-v\d+$", value)]


def test_bilingual_readmes_keep_matching_structure_and_release_facts():
    chinese = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_EN.md").read_text(encoding="utf-8")
    section_pairs = [
        ("## 核心能力", "## Core capabilities"),
        ("## 工作流", "## Workflow"),
        ("## 依赖与安装", "## Requirements and installation"),
        ("## 正确使用方法", "## Correct usage"),
        ("## Checkpoint 与恢复", "## Checkpoints and recovery"),
        ("## 画质建议", "## Quality guidance"),
        ("## 测试", "## Testing"),
        ("## 已知限制", "## Known limitations"),
        ("## 许可证与代码来源", "## License and source attribution"),
    ]
    assert [chinese.index(zh) for zh, _ in section_pairs] == sorted(
        chinese.index(zh) for zh, _ in section_pairs
    )
    assert [english.index(en) for _, en in section_pairs] == sorted(
        english.index(en) for _, en in section_pairs
    )
    assert "[English](README_EN.md)" in chinese
    assert "[简体中文](README.md)" in english
    for fact in (
        "simple / 8", "LoRA strength 1.0", "54.336", "362", "39",
        "13.458", "h3_digital_human_checkpoint_manifest",
        "Run/resume checkpoints → assemble", "video_overlap_frames = 0",
        "270.8", "GPL-3.0-only",
    ):
        assert fact in chinese
        assert fact in english
