# Example inputs

This directory contains the small, redistributable workflow inputs:

- `presenter_reference.png` is a synthetic presenter image generated with an AI
  image service. Its embedded C2PA provenance manifest is intentionally retained.
- `demo_narration.mp3` is an AI-generated demonstration narration track. It
  contains no personal ID3 metadata; only the encoder tag is present.

They are installed with project-prefixed filenames directly under
`ComfyUI/input/` by `scripts/install_example_inputs.py`. This keeps the files
visible to core media widgets. Keep production media out of the repository.

The example files are distributed under GPL-3.0-only with the rest of this
repository. They demonstrate the workflow and do not grant rights to MiniMax,
ComfyUI, or any third-party model used to produce other outputs.
