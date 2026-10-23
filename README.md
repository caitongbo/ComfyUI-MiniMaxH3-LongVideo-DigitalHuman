# ComfyUI MiniMax H3 Long Video Digital Human

[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)

An audio-driven workflow for digital human lip synchronization and long-form video generation with ComfyUI and MiniMax H3.

Generate digital human videos from **input audio and a reference image**, with support for segmented long-audio generation, checkpoint-based resume, clip preview, and video assembly.

[English](README.md) · [简体中文](README.zh-CN.md)

## Preview

**Main workflow:** [MiniMaxH3 - Long Video Digital Human.json](example_workflows/MiniMaxH3%20-%20Long%20Video%20Digital%20Human.json)

![Workflow overview](docs/workflow-overview.png)

See [example outputs and assets](example_outputs/README.md) for the demo video and sample asset details.

Example output: [`digital_human_demo.mp4`](example_outputs/digital_human_demo.mp4)

## Features

- **Audio-driven lip synchronization:** Generate digital human videos using input audio.
- **Segmented long-audio generation:** Automatically determine the number of clips and generate them sequentially.
- **Checkpoint-based resume:** Save per-clip checkpoints and resume interrupted or failed runs.
- **Task controls:** Pause generation, retry failed clips, regenerate from a selected clip, and preserve previous branches.
- **Preview and assembly:** Preview individual clips and assemble previews or final videos from existing clips.
- **Memory-efficient assembly:** Read checkpoints incrementally to reduce memory usage during long-video assembly.

**Generation limit:** Up to 20 consecutive clips. With the default target of approximately 15.08 seconds per clip, the nominal maximum duration is about 4 minutes 31 seconds. Longer content must be split into separate projects.

## Quick Start

### 1. Install

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman.git
cd ComfyUI-MiniMaxH3-LongVideo-DigitalHuman
```

### 2. Check prerequisites

Before running the workflow, prepare the following:

- Python 3.10+
- A ComfyUI version that supports the native MiniMax H3 nodes
- ComfyUI-VideoHelperSuite
- FFmpeg available from the command line
- Required MiniMax H3 model weights and Turbo LoRA

See [Workflow Prerequisites](docs/WORKFLOW_PREREQUISITES.md) for model filenames, installation requirements, and compatibility details.

### 3. Install example assets

The repository includes AI-generated sample assets for testing the workflow.

Run the following command from the repository directory:

```bash
python scripts/install_example_inputs.py
```

To specify a different ComfyUI root directory:

```bash
python scripts/install_example_inputs.py \
  --comfyui-root /path/to/ComfyUI
```

The installer does not overwrite existing files with different contents by default. See the [documentation](docs/README.md) for details.

### 4. Run the workflow

1. Open `example_workflows/MiniMaxH3 - Long Video Digital Human.json` in ComfyUI.
2. Replace the sample reference image and input audio with your own assets.
3. In the `Reusable Clip — MiniMax H3 Reference to Video` node, configure the shared prompt and reference both `<Picture 1>` and `<Picture 2>`. Both image inputs use the same master image node by default.
4. Keep the default generation settings or adjust them for your hardware and quality requirements.
5. Start **Run/resume checkpoints → assemble** from the project assembly node. Do not start the workflow using the regular full-graph Queue action.
6. Pause, retry, preview, or regenerate clips as needed. Finish with **Assemble preview** or **Final export**.

The workflow determines the number of clips automatically from the decoded input audio duration. Manual overrides are not supported.

## Notes and Limitations

- Use a clear, front-facing or near-front-facing medium shot and clean speech audio for best results.
- Using the same master image for both image inputs helps maintain character and scene consistency.
- Character appearance, backgrounds, and fine details may drift during long-form generation.
- Inference uses a 24 fps timeline. The final video is trimmed to the decoded input audio duration.
- Checkpoints may consume significant disk space and are not deleted automatically.
- Do not expose an unauthenticated ComfyUI service directly to untrusted networks.

## Testing

Install the test dependencies and run the checks:

```bash
python -m pip install -r requirements-test.txt
python tests/run_tests.py
python scripts/checksums.py --check
```

CPU-based and static tests do not replace real GPU inference. Before a release, it is recommended to validate multi-clip generation, checkpoint recovery, clip preview, and final video assembly.

## Documentation

- [Documentation Index](docs/README.md)
- [Workflow Prerequisites](docs/WORKFLOW_PREREQUISITES.md)
- [Example Outputs and Assets](example_outputs/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License and Code Attribution

This project is licensed under **GPL-3.0-only**. Some of the initial code originated from the following repositories:

- [seitanism/ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
