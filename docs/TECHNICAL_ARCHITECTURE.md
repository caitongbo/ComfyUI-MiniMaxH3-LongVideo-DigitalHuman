# Technical architecture

## 执行模型

`MiniMaxH3 - Long Video Digital Human.json` 只保存一套 R2V、音频上下文、采样、checkpoint 和预览管线。运行开始时，`MiniMaxH3AudioTimelinePlan` 根据解码后的 waveform 采样数和采样率计算所需 clip 数；前端 manager 随后修改唯一的受管 `clip_index`，把同一恢复门按 Clip 1…N 依次作为 top-level partial prompt 提交。成功后释放模型资源，再进入下一段。

音频规划、每个 clip、最终装配仍是彼此独立的顶层任务。项目没有在一个 Python 节点内循环执行全部采样，因此保留了故障隔离、暂停、重试、断点续跑和防 OOM/Kill 特性。

## 时间线

默认 H3 clip 为 362 帧、24 fps。Clip 1 从主音频 0 秒开始；后续 clip 使用前段 39 帧视觉 latent 作为保护上下文，每段推进 323 帧。音频 conditioning 按绝对 40 Hz 网格切片。最后一段允许尾部补零，assembler 最终按原始主音频精确裁切。

## 连续性与清晰度

唯一母版参考图同时定义身份、服装、人物比例、构图、背景与灯光；同一个 IMAGE 输出连接到 R2V 的 `ref_image_0` 与 `ref_image_1`，在不引入第二份可能冲突素材的前提下重复强化视觉锚点。续段从 manifest 验证 Clip N-1 后直接读取已采样 AV latent，而不是依赖静态前驱图或解码后的 RGB。视觉保护 mask 将继承上下文完全冻结，音频使用对应的绝对时间片。继承前缀不重新加噪，最终装配默认不做 RGB crossfade，避免每个接缝产生二次模糊。

## Checkpoint 协议

每个 clip 独立写入 safetensors。manifest 记录 clip 状态、工作流指纹、take、shape 与 SHA-256。文件先写临时路径，校验后原子发布。后续 clip 依赖前段 lineage；上游变化会使受影响的后继段 stale。

manifest 格式为 `h3_digital_human_checkpoint_manifest`。

## 最终装配

`MiniMaxH3StreamCheckpointDigitalHumanToVHS` 按顺序 mmap 读取 checkpoint，逐段 VAE 解码，并把帧序列流给 VideoHelperSuite。它不构造“所有 clip latent”或“整部 RGB 视频”的常驻内存对象。内存预算在开始前估算，输出文件编号仍由 VHS 管理。

## 浏览器 API

checkpoint 管理端点位于 `/h3_digital_human/checkpoints`，提供状态、项目列表、不可变分支和生成结束后的模型清理。路径始终限制在 ComfyUI output 目录下。
