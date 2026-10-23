# ComfyUI MiniMax H3 Long Video Digital Human

[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)

基于输入音频的数字人口型同步与长视频生成工作流。

使用**输入音频和人物参考图**驱动 MiniMax H3 生成数字人视频，支持长音频分段生成、断点续跑、片段预览与视频装配。

[简体中文](README.md) · [English](README_EN.md) 

## 效果展示

**主工作流：** [MiniMaxH3 - Long Video Digital Human.json](example_workflows/MiniMaxH3%20-%20Long%20Video%20Digital%20Human.json)

![工作流概览](docs/workflow-overview.png)

演示视频与素材说明：[example_outputs/README.md](example_outputs/README.md)

示例成片：[`digital_human_demo.mp4`](example_outputs/digital_human_demo.mp4)

## 核心功能

- **音频驱动口型同步**：使用输入音频生成数字人视频。
- **长音频分段生成**：自动计算片段数量，逐段生成视频。
- **断点续跑**：保存生成检查点，支持失败恢复与任务续跑。
- **任务控制**：支持暂停、失败重试、指定片段重生成及分支保留。
- **预览与装配**：预览单个片段，或使用已有片段生成预览和最终视频。
- **低内存装配**：逐段读取检查点，降低长视频装配的内存占用。

**生成限制：** 最多支持 20 个连续片段，默认每段约 15.08 秒，总时长约 4 分 31 秒。更长内容需要拆分为多个项目。

## 快速开始

### 1. 安装

将仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/caitongbo/ComfyUI-MiniMaxH3-LongVideo-DigitalHuman.git
cd ComfyUI-MiniMaxH3-LongVideo-DigitalHuman
```

### 2. 检查依赖

运行前需要准备：

- Python 3.10+
- 支持 MiniMax H3 原生节点的 ComfyUI
- ComfyUI-VideoHelperSuite
- FFmpeg
- MiniMax H3 所需模型权重及 Turbo LoRA

模型文件、安装要求与兼容性说明见[工作流运行前提](docs/WORKFLOW_PREREQUISITES.md)。

### 3. 安装示例素材

仓库提供 AI 生成的测试人物参考图和示例音频，可用于验证工作流。

```bash
python scripts/install_example_inputs.py
```

如需指定 ComfyUI 根目录：

```bash
python scripts/install_example_inputs.py \
  --comfyui-root /path/to/ComfyUI
```

脚本默认不会覆盖内容不同的同名文件。详细说明见[示例素材安装说明](docs/README.md)。

### 4. 运行工作流

1. 在 ComfyUI 中打开 `example_workflows/MiniMaxH3 - Long Video Digital Human.json`。
2. 将示例人物参考图和输入音频替换为自己的素材。
3. 在 `Reusable Clip — MiniMax H3 Reference to Video` 节点配置共享 Prompt，并引用 `<Picture 1>` 和 `<Picture 2>`。两个图像输入默认来自同一个母版图节点。
4. 保持工作流默认生成参数，或根据硬件与画质需求调整。
5. 从项目装配节点启动 **Run/resume checkpoints → assemble**，不要使用普通全图 Queue 启动任务。
6. 按需暂停、重试、预览或重新生成片段，最后选择 **Assemble preview** 或 **Final export**。

片段数量根据解码后的输入音频时长自动计算，不支持手动覆盖。

## 注意事项

- 建议使用清晰的正面或近正面中景人物参考图，以及干净的人声音频。
- 同一张母版图用于两个图像输入，有助于保持人物与场景一致。
- 长视频生成可能出现人物外观、背景或细节漂移。
- 推理采用 24 fps 时间基准，最终视频会裁切至输入音频的解码时长。
- 检查点可能占用较多磁盘空间，不会自动清理。
- 不要将未受认证保护的 ComfyUI 服务直接暴露给不受信任的网络。

## 测试

安装测试依赖并运行检查：

```bash
python -m pip install -r requirements-test.txt
python tests/run_tests.py
python scripts/checksums.py --check
```

CPU 和静态测试不能替代真实 GPU 推理。发布前建议验证多个片段的生成、断点恢复、片段预览与最终视频装配。

## 详细文档

- [文档索引](docs/README.md)
- [工作流运行前提](docs/WORKFLOW_PREREQUISITES.md)
- [示例输出与素材说明](example_outputs/README.md)
- [贡献指南](CONTRIBUTING.md)
- [安全报告](SECURITY.md)
- [更新记录](CHANGELOG.md)

## 许可证与代码来源

本项目按 GPL-3.0-only 许可证发布。部分初始代码历史来自：

- [seitanism/ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
