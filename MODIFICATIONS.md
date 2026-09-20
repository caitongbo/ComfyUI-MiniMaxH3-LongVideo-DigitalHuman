# Modifications

## Digital-human specialization

- 项目发行名调整为 `ComfyUI-MiniMaxH3-LongVideo-DigitalHuman`。
- 示例目录收敛为唯一的 `MiniMaxH3 - Long Video Digital Human.json` 工作流。
- 默认工作流改为长音频、最多 20 clips、约 15 秒/clip 的数字人路径。
- 由解码音频自动计算 clip 数；20 套静态生成链收敛为一套可复用 R2V / Context / Sampler / Checkpoint 管线和一个共享 Prompt。
- BASE 使用原生 Turbo LoRA 8 步，统一生成速度、人物稳定性与口型细节的默认质量路径。
- 新增逐 clip 原子 checkpoint、manifest、恢复、重试、暂停、分支和从指定片段重生成。
- 新增 mmap/disk-backed 最终装配，避免长视频最后合成阶段被系统 OOM kill。
- checkpoint/API/output 命名统一到 `h3_digital_human`。
- 节点、前端调度、checkpoint/API/output 与日志前缀统一为 `DigitalHuman` 命名；发行代码只保留该工作流实际使用的运行路径。
- manifest 格式统一为无版本后缀的 `h3_digital_human_checkpoint_manifest`；不保留旧格式兼容。
- 唯一工作流同时提供数字人长视频生成、恢复、预览装配和最终导出。

## Upstream

本项目基于 `seitanism/ComfyUI-H3-Motion-Context-MultiRef`，后者基于 `NikoDemon80/ComfyUI-H3-Motion-Context`。原有贡献与 GPL-3.0 许可证继续保留。
