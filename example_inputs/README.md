# Example inputs

The files in `h3_digital_human/` are small, redistributable workflow inputs:

- `presenter_reference.png` is a synthetic presenter image generated with an AI
  image service. Its embedded C2PA provenance manifest is intentionally retained.
- `demo_narration.mp3` is an AI-generated demonstration narration track. It
  contains no personal ID3 metadata; only the encoder tag is present.

They are installed into `ComfyUI/input/h3_digital_human/` by
`scripts/install_example_inputs.py`. Keep production media out of the repository.

The example files are distributed under GPL-3.0-only with the rest of this
repository. They demonstrate the workflow and do not grant rights to MiniMax,
ComfyUI, or any third-party model used to produce other outputs.
