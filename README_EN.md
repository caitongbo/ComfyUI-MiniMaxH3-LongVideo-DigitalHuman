# ComfyUI MiniMax H3 Long Video Digital Human

Checkpoint-managed long-form digital-human video generation for MiniMax H3 in
ComfyUI. The project generates one clip at a time from a master audio track,
saves each joint audio/video latent atomically, resumes interrupted runs, and
assembles the final video with bounded memory use.

[中文说明](README.md)

## Highlights

- Up to 20 sequential clips; approximately 15.08 seconds for the first clip and
  13.46 seconds of new timeline coverage for each continuation.
- Exact decoded-audio duration automatically selects the clip count; one reusable
  generation pipeline and one shared prompt replace 20 duplicated static chains.
- Native Turbo LoRA defaults: `simple`, 8 steps, denoise 1, LoRA strength 1.0.
- Verified safetensors checkpoints with manifest, lineage, hashes, retries,
  pause/resume, branching, and regeneration from a selected clip.
- Direct latent-tail continuation and drift control for long-run identity stability.
- Checkpoint-selectable clip preview and disk-backed final assembly trimmed to master-audio length.
- One supported workflow for generation, recovery, checkpoint preview assembly,
  and final export.

## Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman.git
cd ComfyUI-MiniMaxH3-LongVideo-DigitalHuman
python scripts/install_example_inputs.py
```

Restart ComfyUI. The workflow requires a current ComfyUI with native MiniMax H3
nodes, ComfyUI-VideoHelperSuite, the MiniMax H3 model components listed in
[WORKFLOW_PREREQUISITES.md](WORKFLOW_PREREQUISITES.md), and FFmpeg.

Recorded workflow selections (weights are not distributed):

- diffusion: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- video VAE: `minimax_h3_video_vae_fp16.safetensors`
- audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- Turbo LoRA: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

## Quick start

The bundled `presenter_reference.png` and `demo_narration.mp3` are both
AI-generated and do not represent a real person or recording. Their stable,
purpose-based names intentionally do not encode the current duration or clip
count.

1. Open `example_workflows/MiniMaxH3 - Long Video Digital Human.json`.
2. Replace the example image/audio for production and edit the single reusable
   R2V prompt. `Auto from Audio` detects that the bundled 54.336-second demo needs four clips.
3. Keep the base sampler at `simple / 8 / denoise 1` with Turbo LoRA strength 1.0.
4. Start from **Run/resume checkpoints → assemble** on the project assembler.
   Do not queue the entire graph as a normal full prompt.
5. Use **Preview clip** in the project panel, then use **Final export**.

Two distinct reference images are optional. The demo reuses one portrait in the
face and body/wardrobe slots. If available, use mutually consistent close-up and
half/full-body references of the same person and styling.

## Validation and support

```bash
python -m pip install -r requirements-test.txt
python tests/run_tests.py
python scripts/checksums.py --check
```

CPU/static tests do not replace GPU inference testing. See
[CONTRIBUTING.md](CONTRIBUTING.md) for actionable bug reports and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.

Known limits: 20 clips (about 270.8 seconds of timeline coverage), fixed 24 fps
H3 timing, non-zero long-run identity drift risk, and potentially substantial
checkpoint disk usage. Never expose an unauthenticated ComfyUI server directly
to an untrusted network.

## License

GPL-3.0-only. See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md), and
[MODIFICATIONS.md](MODIFICATIONS.md). This is an independent community project,
not an official MiniMax or Comfy Org release.
