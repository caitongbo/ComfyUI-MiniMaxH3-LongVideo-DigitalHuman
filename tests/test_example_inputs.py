from pathlib import Path

import pytest

from scripts.install_example_inputs import ASSETS, SOURCE_DIR, install


def test_installer_copies_bundled_assets_to_namespaced_comfy_input(tmp_path: Path):
    (tmp_path / "custom_nodes").mkdir()
    installed = install(tmp_path)

    assert installed == [tmp_path / "input" / "h3_digital_human" / name for name in ASSETS]
    for source_name, target in zip(ASSETS, installed):
        assert target.read_bytes() == (SOURCE_DIR / source_name).read_bytes()


def test_installer_does_not_overwrite_different_user_file_without_force(tmp_path: Path):
    (tmp_path / "custom_nodes").mkdir()
    target_dir = tmp_path / "input" / "h3_digital_human"
    target_dir.mkdir(parents=True)
    protected = target_dir / ASSETS[0]
    protected.write_bytes(b"user-owned")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install(tmp_path)
    assert protected.read_bytes() == b"user-owned"
