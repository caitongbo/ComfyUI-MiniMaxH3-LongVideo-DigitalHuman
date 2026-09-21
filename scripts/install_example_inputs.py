"""Install the bundled digital-human example media into ComfyUI/input."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "example_inputs"
# Core media widgets can falsely mark files in input subdirectories as missing.
# Install collision-resistant filenames directly under input/ so both the
# frontend selector and backend loader resolve the saved workflow value.
ASSETS = (
    ("presenter_reference.png", "h3_digital_human_presenter_reference.png"),
    ("demo_narration.mp3", "h3_digital_human_demo_narration.mp3"),
)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _default_comfyui_root() -> Path | None:
    custom_nodes = PROJECT_ROOT.parent
    if custom_nodes.name == "custom_nodes":
        return custom_nodes.parent
    return None


def install(comfyui_root: Path, *, force: bool = False) -> list[Path]:
    comfyui_root = comfyui_root.expanduser().resolve()
    if not comfyui_root.is_dir():
        raise ValueError(f"ComfyUI root does not exist: {comfyui_root}")
    if not ((comfyui_root / "main.py").is_file() or
            (comfyui_root / "custom_nodes").is_dir()):
        raise ValueError(f"Not a ComfyUI root (main.py/custom_nodes missing): {comfyui_root}")

    destination = comfyui_root / "input"
    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    for source_name, target_name in ASSETS:
        source = SOURCE_DIR / source_name
        target = destination / target_name
        if not source.is_file():
            raise FileNotFoundError(f"Bundled example asset is missing: {source}")
        if target.exists() and _digest(target) != _digest(source) and not force:
            raise FileExistsError(
                f"Refusing to overwrite a different file: {target}. "
                "Use --force to replace it."
            )
        if not target.exists() or _digest(target) != _digest(source):
            shutil.copy2(source, target)
        installed.append(target)
    return installed


def main() -> None:
    detected = _default_comfyui_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfyui-root",
        type=Path,
        default=detected,
        help="ComfyUI root; auto-detected when this project is under custom_nodes/",
    )
    parser.add_argument("--force", action="store_true", help="replace different existing example files")
    args = parser.parse_args()
    if args.comfyui_root is None:
        parser.error("cannot auto-detect ComfyUI; pass --comfyui-root /path/to/ComfyUI")
    for path in install(args.comfyui_root, force=args.force):
        print(path)
    print("Refresh the ComfyUI browser page; restart ComfyUI first if the media nodes remain red.")


if __name__ == "__main__":
    main()
