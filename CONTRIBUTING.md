# Contributing

Thanks for helping improve the project. Keep contributions focused on the
published MiniMax H3 digital-human workflow, checkpoint lifecycle, continuity,
and low-memory assembly path.

## Before opening an issue

- Use the current `main` branch and a current ComfyUI installation.
- Reproduce with the bundled workflow before testing a modified graph.
- Do not attach model weights, private media, API keys, or checkpoint files.
- For generation failures, include the complete traceback, ComfyUI commit,
  operating system, Python/PyTorch versions, GPU/VRAM, model filenames, timeline
  mode, planned clip count, and the first relevant `h3_digital_human` log lines.
- Report vulnerabilities according to [SECURITY.md](SECURITY.md), not in a public issue.

## Development checks

Create an isolated environment and install the CPU/static test dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
python tests/run_tests.py
python scripts/checksums.py --check
```

Tests intentionally run each module in a fresh process because their ComfyUI
mocks must not leak between modules. GPU inference is a separate release check.

## Pull requests

- Keep unrelated formatting and refactors out of the change.
- Add or update focused regression tests.
- Update README, architecture, prerequisites, and changelog when behavior changes.
- Run the complete test suite and regenerate `SHA256SUMS.txt` last.
- Keep runtime code limited to features used by the supported workflow; new
  public nodes require a supported workflow, documentation, and tests.
- By contributing, you agree that your contribution is distributed under GPL-3.0-only.
