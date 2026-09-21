# Modifications and upstream attribution

This distribution specializes the upstream motion-context implementation for
one supported MiniMax H3 digital-human workflow. Its principal modifications
are:

- automatic clip planning from decoded master-audio duration;
- one reusable Ref2VA/context/sampler/checkpoint pipeline and one shared prompt;
- one master image connected to both image-reference slots;
- native Turbo LoRA 8-step defaults;
- atomic per-clip checkpoints with manifest, hashes, lineage, recovery, retry,
  branching, and regeneration controls;
- disk-backed final decoding and assembly with exact audio-length trimming;
- a focused five-node public surface, browser-side safe scheduling, and
  project-specific checkpoint routes; and
- names, paths, examples, tests, and documentation scoped to
  `h3_digital_human`.

## Upstream

Modified code in this repository originates from
[seitanism/ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef),
which is based on
[NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context).
Their copyright notices and the GPL-3.0-only license are retained.
