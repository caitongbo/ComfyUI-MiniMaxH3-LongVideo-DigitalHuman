# Release checklist

## One-time repository setup

1. Create or rename the GitHub repository to
   `caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman`.
2. Enable **Settings → Security → Private vulnerability reporting** so the link
   in `SECURITY.md` and the issue chooser works.
3. Set the repository description, GPL-3.0 license, and topics such as `comfyui`,
   `minimax-h3`, `digital-human`, `long-video`, and `video-generation`.
4. Protect `main`: require the CI check, disallow force pushes, and require review
   for external pull requests.
5. Confirm that the maintainer has the right to redistribute both files under
   `example_inputs/` under the license stated in `NOTICE.md`.
6. Create the `caitongbo` publisher in the Comfy Registry, or replace
   `PublisherId` in `pyproject.toml` with the exact registered publisher ID.

## Every release

1. Update the version in `pyproject.toml` and `__init__.py`.
2. Move entries from `Unreleased` into a dated section in `CHANGELOG.md`.
3. Run:

   ```bash
   python tests/run_tests.py
   python scripts/checksums.py
   python scripts/checksums.py --check
   git diff --check
   ```

4. Perform a real GPU smoke test: four demo clips, interrupted resume, one
   regeneration from the middle, per-clip preview, and final checkpoint assembly.
5. Review the staged file list for checkpoints, weights, private media, absolute
   paths, logs, and secrets.
6. Merge to `main`, create a signed `vX.Y.Z` tag, and publish GitHub release notes.
7. Optionally publish the same `pyproject.toml` version to the Comfy Registry with
   `comfy-cli publish`; keep the registry token only in GitHub Actions secrets.
