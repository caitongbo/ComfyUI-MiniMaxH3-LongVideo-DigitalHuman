# ComfyUI MiniMax H3 Long Video Digital Human

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A ComfyUI workflow and recovery toolkit for long-form MiniMax H3 digital-human
videos. The project centers on one workflow: it turns long audio into sequential
clips, publishes each clip atomically, resumes interrupted runs, and assembles
the final video with bounded memory use.

Repository and recommended installation directory:
`ComfyUI-MiniMaxH3-LongVideo-DigitalHuman`.

[简体中文](README.md) · **English** · [Documentation](docs/README.md) · [Issues](https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman/issues) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

> This is an independent community project, not an official MiniMax or Comfy
> Org release. Model weights are not distributed with this repository.

## Core capabilities

- Drive up to 20 sequential clips directly from long audio; the default clip is
  approximately 15.08 seconds with 39 frames of visual context.
- Determine clip count automatically from decoded audio duration while reusing
  one generation pipeline and one prompt.
- Use native Turbo LoRA BASE defaults: `simple / 8 steps / denoise 1 / LoRA strength 1.0`.
- Save every clip independently as `.safetensors`; the manifest verifies
  parameters, order, lineage, and file hashes.
- Schedule one clip at a time with retry, pause, regeneration from a selected
  clip, and immutable project branches.
- Decode checkpoints incrementally with mmap during final assembly instead of
  keeping all samplers or the complete RGB movie in memory.
- Preview any individual clip and trim the final video to the exact master-audio
  duration.
- Continue after Clip 1 by inheriting and fully freezing the previous raw latent
  tail, without re-noising the protected prefix.

## Workflow

[`MiniMaxH3 - Long Video Digital Human.json`](example_workflows/MiniMaxH3%20-%20Long%20Video%20Digital%20Human.json)
is the only supported workflow. It covers generation, recovery, individual clip
preview, checkpoint preview assembly, and final export. After restarting
ComfyUI, reopen this workflow to resume generation or assemble verified
checkpoints directly.

See [Demo video](example_outputs/README.md) for the generated example and file
requirements. The showcase is stored at
`example_outputs/digital_human_demo.mp4` and excluded from the Comfy Registry
installation package.

## Requirements and installation

Clone the project under `ComfyUI/custom_nodes/`, then restart ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman.git
cd ComfyUI-MiniMaxH3-LongVideo-DigitalHuman
```

The repository includes two inputs for validating the workflow:

- `example_inputs/presenter_reference.png`: a 940 × 1672
  portrait presenter reference;
- `example_inputs/demo_narration.mp3`: a 54.336-second,
  24 kHz mono narration sample.

Both assets are AI-generated and do not represent a real person or recording.
Their stable purpose-based names do not encode the current duration or default
clip count.

Install them into ComfyUI's standard input directory before first use:

```bash
cd ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman
python scripts/install_example_inputs.py
```

If the project is not under `custom_nodes/`, specify the ComfyUI root:

```bash
python scripts/install_example_inputs.py --comfyui-root /path/to/ComfyUI
```

The script installs the examples as
`ComfyUI/input/h3_digital_human_presenter_reference.png` and
`ComfyUI/input/h3_digital_human_demo_narration.mp3`. Project-prefixed names
avoid collisions, while placing them directly under `input/` prevents the
ComfyUI frontend from falsely reporting subdirectory media as unselected.
Refresh the browser after installation; if ComfyUI was already running and the
nodes remain red, restart ComfyUI before reopening the workflow. The installer
refuses to overwrite different files; use `--force` only to restore the
bundled examples.

Runtime requirements:

- Python 3.10 or newer;
- a recent ComfyUI release with native MiniMax H3 nodes;
- `ComfyUI-VideoHelperSuite`, which supplies `VHS_VideoCombine` for clip preview
  and final H.264 output;
- FFmpeg available on the command line; and
- the MiniMax H3 model weights listed below.

The workflow records these default model filenames, which can be changed to
match the local installation:

- diffusion: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- video VAE: `minimax_h3_video_vae_fp16.safetensors`
- audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- Turbo LoRA: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

See [Workflow prerequisites](docs/WORKFLOW_PREREQUISITES.md) for the detailed
checklist.

The package registers only the five nodes used by the supported workflow, all
displayed under `H3 Digital Human —`. No other MiniMax H3 custom-node repository
is required beyond ComfyUI and VideoHelperSuite.

## Correct usage

1. Install the example inputs and open
   `example_workflows/MiniMaxH3 - Long Video Digital Human.json`. Replace the
   bundled reference and audio for production. The master image should define
   identity, wardrobe, composition, background, and lighting together.
2. Edit the one shared prompt in
   `Reusable Clip — MiniMax H3 Reference to Video` so it refers to both
   `<Picture 1>` and `<Picture 2>`. Both inputs come from the same master-image
   node; do not upload a second image or duplicate the prompt per clip.
3. `AUTO TIMELINE` always calculates clip count from the actual decoded sample
   count. There is no manual clip-count override; the project panel displays
   `Not planned` before planning runs.
4. Keep BASE at `simple / 8 / denoise 1` with Turbo LoRA strength `1.0`. Start
   from **Run/resume checkpoints → assemble** on the project assembler instead
   of using a normal full-graph Queue.
5. The manager first runs lightweight audio planning and then submits the same
   generation pipeline once per clip, releasing models between clips. Use
   **Pause after current clip**, **Regenerate from here**, **Preview clip**, and
   automatic retry as needed; finish with **Assemble preview** or
   **Final export**.

The frontend redirects a normal Queue to safe scheduling. If ComfyUI reports
`Use 'Run/resume checkpoints → assemble'`, the safety extension did not
intercept that queue attempt. Refresh the browser, confirm that this project's
JavaScript loaded, and start from the checkpoint assembler.

### Duration and clip count

The default target is 362 frames, or `362 / 24 = 15.083 s`. Starting with Clip
2, 39 context frames overlap the preceding clip, so the timeline advances by
`323 / 24 = 13.458 s` per continuation.

Examples:

- 12 seconds: 1 clip, trimmed to 12 seconds during final assembly;
- 40 seconds: 3 clips covering about 42 seconds, trimmed to 40 seconds;
- bundled 54.336-second audio: 4 clips covering about 55.46 seconds, then
  trimmed exactly to the decoded audio length; and
- when the final audio slice is shorter than the H3 generation grid, only the
  conditioning tail is zero-padded. The final movie contains no extra silence.

### Individual clip preview

The workflow contains one `Current Managed Clip — VHS Preview`. Select
**Preview clip** for a clip in the project panel; the manager points the reusable
index at that checkpoint and submits only the preview node. Preview is for
reviewing a take. Use the checkpoint assembler for the final export.

## Checkpoints and recovery

Default paths:

- BASE: `ComfyUI/output/h3_digital_human/checkpoints/clip_*.safetensors`
- manifest: `clip.manifest.json` in the same directory

The manifest format is `h3_digital_human_checkpoint_manifest`.

Each checkpoint is written to a temporary file and atomically replaced after
validation. Reopen the workflow after a restart and select Run/Resume to
continue from the first missing, damaged, or stale clip. Changing an upstream
model, prompt, sampler setting, audio track, or reference image invalidates
affected clips. **Regenerate from here** explicitly removes that clip's record
and all successor records. Use **Branch here…** when the earlier take must be
preserved.

The assembler always verifies the current manifest, SHA-256, and predecessor
lineage. There is no bypass switch. When reopening a project, keep the
checkpoint prefix, VAE, timeline parameters, and master audio identical to the
generation run before using **Assemble preview** or **Final export**.

## Quality guidance

- Use native Turbo 8-step BASE for the intended balance of generation speed,
  presenter stability, and lip detail.
- Upload one approved medium-shot master image containing a clear face, target
  wardrobe, body proportions, composition, background, and lighting. The
  workflow sends it to both `<Picture 1>` and `<Picture 2>`: the first anchors
  identity, while the second reinforces wardrobe, framing, and scene. Do not
  replace the second input with a separate face close-up whose crop or head
  scale can leak into the output.
- If continuation quality, identity, or background drifts, do not re-noise the
  inherited prefix or add an RGB crossfade. The default
  `video_overlap_frames = 0` avoids repeatedly blurring seams.
- Prefer a sharp frontal reference, clean speech, and one stable prompt. A face
  that occupies too little of the source image directly limits mouth and facial
  detail.

## Testing

```bash
python -m pip install -r requirements-test.txt
python tests/run_tests.py
python scripts/checksums.py --check
```

CPU/static tests do not replace real GPU inference. Before a release, validate
at least two BASE clips, interrupted recovery, individual clip preview, and
final assembly.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance. Never upload
private media, model weights, checkpoints, or access tokens to a public issue;
report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## Known limitations

- Maximum 20 clips; split new timelines longer than approximately 270.8 seconds
  into separate projects.
- H3 inference uses a fixed 24 fps timebase; final duration is determined by
  trimming to the decoded master audio.
- Long-run identity and detail drift can be reduced but cannot be guaranteed to
  disappear. Conflicting reference information makes drift more visible.
- Checkpoints can consume substantial disk space and are never deleted
  automatically.
- ComfyUI exposes local file and checkpoint-management capabilities; never
  expose an unauthenticated server directly to an untrusted network.

## License and source attribution

This project is distributed under GPL-3.0-only. Part of its initial code history
comes from
[seitanism/ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)
and
[NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context),
so their contribution attribution and the GPL-3.0 license are retained. This
source notice does not mean those repositories are runtime dependencies. See
[LICENSE](LICENSE), [NOTICE.md](NOTICE.md), and
[MODIFICATIONS.md](MODIFICATIONS.md) for details.
