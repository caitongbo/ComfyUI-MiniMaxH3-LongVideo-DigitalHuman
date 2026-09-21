# Workflows

本目录只提供一个入口：

- `MiniMaxH3 - Long Video Digital Human.json`：数字人长音频 → 长视频的完整生成、恢复、单段预览、checkpoint 预览装配与最终导出工作流。

首次运行先在项目根目录执行 `python scripts/install_example_inputs.py`。它会把发行源
`example_inputs/` 安装为 ComfyUI `input/` 根目录中的
`h3_digital_human_presenter_reference.png` 与 `h3_digital_human_demo_narration.mp3`；工作流已默认引用这两个文件。安装后刷新浏览器，必要时重启 ComfyUI，再打开工作流。

主工作流的日常操作顺序：安装或选择一张母版参考图和音频 → 填写同时引用 `<Picture 1>` 与 `<Picture 2>` 的唯一 Prompt → 从项目装配节点点击 `Run/resume checkpoints → assemble` → 验收单段预览与最终成片。两路 Picture 均来自同一个图像节点；规划节点始终根据解码音频自动计算 clips，随附示例音频需要 4 clips，不需要也不能手工覆盖。

工作流只需上传一张母版参考图，但会将其同时连接到 R2V 的前两个参考槽，以重复强化身份、服装、人物比例、构图、背景和灯光。不要把第二路改接独立近脸图，以免近景裁切和头部比例影响生成画面。

不要直接执行整张图。工作流只有一套可复用 R2V / Context / Sampler / Checkpoint 管线；管理器先测量音频，再用不同 `clip_index` 将同一管线逐段提交。每段仍是独立顶层任务，因此不会把长视频的所有 sampler 同时塞进一个 prompt。

ComfyUI 重启后重新打开同一个工作流，即可继续生成，或通过 **Assemble preview** / **Final export** 直接装配已有 checkpoint。prefix、主音频、帧参数和 VAE 必须与生成项目一致；manifest、SHA-256 和前驱 lineage 校验始终强制执行。项目面板的 **Preview clip** 使用唯一预览节点查看指定 checkpoint。
