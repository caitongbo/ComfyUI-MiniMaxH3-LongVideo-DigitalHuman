# Workflow prerequisites

## 必需

- 较新的 ComfyUI，包含 MiniMax H3 Ref2VA、H3 音视频 VAE 和 masked velocity 所需能力。
- `ComfyUI-VideoHelperSuite`。
- 足够的模型磁盘空间，以及可写的 `ComfyUI/output/`。
- 工作流所引用的 diffusion、CLIP、video VAE、audio VAE 与 Turbo LoRA。

工作流保存时记录的核心节点 schema 为 `0.30.0`–`0.33.0`，VideoHelperSuite 节点为 `1.7.7`。这些是工作流节点记录值，不是严格的包版本下限；公开发布默认支持当前稳定版 ComfyUI 与当前 VideoHelperSuite。报告兼容问题时必须附上实际 ComfyUI commit 和扩展版本。

## 启动检查

1. 运行 `python scripts/install_example_inputs.py`，确认 `ComfyUI/input/h3_digital_human/` 中存在 `presenter_reference.png` 和 `demo_narration.mp3`。
2. 浏览器加载后能看到 `H3 Digital Human — Audio Timeline Plan` 与 checkpoint 管理按钮，音频和参考图加载节点没有红色缺失提示。
3. BASE prefix 为 `h3_digital_human/checkpoints/clip`。
4. BASE scheduler 是 `simple / 8 / 1`，Turbo LoRA 强度为 `1.0`。
5. `AUTO TIMELINE` 默认为 `Auto from Audio`；规划输出直接连接 assembler 的 `clip_count`。
6. 最终 assembler 使用 `disk-backed` 和 `trim_to_audio = true`；manifest 校验为强制行为。

## 示例素材约定

仓库中的发行源固定放在 `example_inputs/h3_digital_human/`，运行时副本固定放在 `ComfyUI/input/h3_digital_human/`。工作流引用运行时相对路径，不直接读取 `custom_nodes` 目录，也不记录绝对路径。

示例音频长 54.336 秒，自动规划结果为 4 clips。首次只想检查依赖和显存时，可把规划模式临时切到 `Manual` 并设置 1；继续生成或重新装配时，必须恢复原时间线设置并选择生成 checkpoints 时使用的同一条主音频。

示例参考图本身同时具备清晰脸部、上半身服装和目标演播室构图，所以参考图 1、2 默认复用同一文件，不要求额外准备第二张图片。如果另有同一人物、同一造型的近脸图，可放入参考图 1，并把当前半身图保留在参考图 2；两张图的身份、发型、服装或场景冲突时，单张高质量参考通常更稳定。

## 常见非阻塞日志

ComfyUI-Manager 获取 `custom-node-list.json` 或 `github-stats.json` 超时只影响 Manager 在线目录刷新，不代表 H3 生成失败。只要 sampler 完成并发布 checkpoint，即可继续。

## 最小 GPU 验证

先用 `Manual = 2` 验证以下链路：自动索引 Clip 1 → 从磁盘加载已验证前驱并生成 Clip 2 → 重启/刷新后复用 verified checkpoint → 项目面板 `Preview clip` → 最终装配。CPU 测试无法验证显存峰值或 VAE 解码。
