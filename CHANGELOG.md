# Changelog

All notable user-facing changes will be documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Install bundled example media directly under `ComfyUI/input/` with
  project-prefixed filenames, avoiding false missing-media errors in the
  ComfyUI frontend for files stored in input subdirectories.

## [1.0.0] - 2026-09-22

- Initial public release focused on MiniMax H3 long-video digital humans.
- Added decoded-audio timeline planning for up to 20 sequential clips with one
  reusable generation pipeline and one shared prompt.
- Added atomic safetensors checkpoints with hashes, lineage validation,
  pause/resume, retries, branching, regeneration from a selected clip, and
  checkpoint-selectable preview.
- Added disk-backed final assembly and exact trimming to the decoded master
  audio length.
- Added a single-master-image workflow, native Turbo LoRA 8-step defaults, and
  a structured Ref2VA presenter prompt.
- Added AI-generated example inputs, bilingual documentation, release checks,
  and CPU/static regression tests.

[Unreleased]: https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman/releases/tag/v1.0.0
