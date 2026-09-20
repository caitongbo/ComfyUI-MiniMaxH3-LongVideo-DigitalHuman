# Changelog

All notable user-facing changes are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Replaced 20 duplicated static Clip chains with one reusable, checkpoint-driven
  generation pipeline and one shared prompt.
- Added decoded-audio timeline planning with automatic clip-count calculation,
  manual override, managed clip indexing, and checkpoint-selectable preview.

## [1.0.0] - 2026-09-20

- Focused the package on checkpoint-managed MiniMax H3 digital-human videos.
- Added sequential generation for up to 20 clips with resume, retry, branching,
  stale-lineage detection, and atomic safetensors checkpoints.
- Added disk-backed final assembly to avoid keeping every latent or decoded frame
  in memory at once.
- Added exact master-audio slicing, final audio-length trimming, latent-tail
  continuation, drift control, per-clip previews, and a four-clip demo workflow.
- Added bundled example inputs and a non-destructive installer.

[Unreleased]: https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman/releases/tag/v1.0.0
