# Workflows

本目录只提供一个入口：

- `MiniMaxH3 - Long Video Digital Human.json`：数字人长音频 → 长视频的完整生成、恢复、单段预览、checkpoint 预览装配与最终导出工作流。

首次运行先在项目根目录执行 `python scripts/install_example_inputs.py`。它会把发行源 `example_inputs/h3_digital_human/` 复制到 ComfyUI 标准输入目录 `input/h3_digital_human/`；工作流已默认引用其中的 `presenter_reference.png` 和 `demo_narration.mp3`。

主工作流的日常操作顺序：安装或选择参考图和音频 → 填写唯一 Prompt → 保持 `Auto from Audio` → 从项目装配节点点击 `Run/resume checkpoints → assemble` → 验收单段预览与最终成片。规划节点会识别随附示例音频需要 4 clips，不需要手工计算。

两张不同的参考图不是硬性要求。示例图同时覆盖脸部、服装和演播室构图，因而在参考图 1、2 中复用；若有同一人物、同一造型的独立近脸图与半身/全身图，再分别替换会更有利于角色身份和造型约束。

不要直接执行整张图。工作流只有一套可复用 R2V / Context / Sampler / Checkpoint 管线；管理器先测量音频，再用不同 `clip_index` 将同一管线逐段提交。每段仍是独立顶层任务，因此不会把长视频的所有 sampler 同时塞进一个 prompt。

ComfyUI 重启后重新打开同一个工作流，即可继续生成，或通过 **Assemble preview** / **Final export** 直接装配已有 checkpoint。prefix、主音频、帧参数和 VAE 必须与生成项目一致；manifest、SHA-256 和前驱 lineage 校验始终强制执行。项目面板的 **Preview clip** 使用唯一预览节点查看指定 checkpoint。
