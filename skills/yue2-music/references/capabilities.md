# 全功能映射与覆盖依据

核对日期 2026-09-14，来源以当前项目源码为准，README 不一致时以 parser/protocol 优先。

| 能力 | 实际入口 | 技能路径 / 验证 |
|---|---|---|
| full 自动旋律和弦谱 + 音乐 | generate / pipe.plan+其余三阶段 | generate_music --action generate |
| full supplied ABC | generate, cot=full+abc | 原生 ABC preflight；请求不重写谱 |
| melody 自动谱 | generate, cot=melody | same helper |
| melody supplied | generate, cot=melody+abc | chord-free 校验/strip-chords |
| off 直接生成 | generate, cot=off | 禁 ABC；仍可人声 |
| sheet-only | plan | generate_music --action plan；原始 ID/manifest |
| exact saved plan | render-plan | generation-and-covers |
| 再解码 / NAR 重放 | replay --stage decode/synthesize | 保存 config 和噪声，不靠旧音频改词 |
| 8/32 任意正整数步对比 | GenerationConfig.ode_steps | compare_steps，same-input provenance |
| AR sampling / semantic CFG | abc_sampling/semantic_sampling/cfg_scale | 原生区间校验和单变量试验 |
| BF16 / 8bit / 4bit AR | prepare + --precision | 4bit 实验性，缺失不回退 |
| 音频转谱 | transcribe full/melody-full/melody-vocal | 本地 SheetSage2 + MERT2；default/paper preset |
| 多窗口转谱 | transcription.pipeline.transcribe | 整曲输入；--max-seconds 仅显式诊断 |
| 转谱导出 | ABC/MIDI/LAB/结构化 JSON | result 和原始转谱保留；非人声 ASR |
| 音频端到端 cover | cover --audio SOURCE REQUEST | generate_music --action cover；完整词需请求提供 |
| 和声/音高/节奏/tempo/meter/结构编辑 | 新 ABC + generate | editing-workflows + abc-editing + inspect/compare |
| 去和弦/选声部 | abc strip-chords --keep-voice | 保留 rests/声部时值 |
| 歌词改写/多语言演唱 | lyrics + 新请求 | 六语言历史实例不等于任意语言质量保证 |
| 辅助创作/歌词-only | agent 写作 | 不触发生成；style/lyrics/选用模式草案 |
| 音乐特征提取 | lyra.mert.MERT2(config) / convert_mert_weights | read src/lyra/mert.py 按当前签名；不是 codec/ASR |
| 串行批量与复用 | batch --concurrency 1 --resume | 合法 casefold unique id，完整结果身份匹配 |
| 中断/回调/进度 | cancelled / on_token / progress | Python API；资源失败关闭后新 attempt |
| 离线转换/导出/检查 | prepare / save_pretrained / doctor | setup；本地权重，显式下载/复制 |
| 自定义 tiling/资源限制 | query_chunk_size / vae_core_frames / budget | 原生 full attention，guard 常开 |
| FLAC/WAV 与产物/哈希 | SongResult.save/save_artifacts | summary 校验、保留 noise/latent/配置 |
| 本地统一试听 / demo 基准 | listen / 项目 benchmark tools | listening-and-evaluation |

MERT2 是底层模块，不是 from_pretrained 特征 CLI：按 transcription/model.py 的加载器
加载转换权重，调用 MERT2(config)(waveform, layer_weights=None, cancelled=None)；
转谱前端使用 24 kHz mono，遵守 checkpoint config。特征不是语义 codec IDs。
可选 legacy VAE 需显式路径和独立产物，不能代替默认 VAE 来掩盖差异。

不提供：训练/LoRA、流式实时音频、波形 inpainting、换声克隆、分轨/锁伴奏、硬指定
音频时长、音素/词级时间戳、MIDI 直接作为生成请求（先明确转为支持的 ABC）。
不要把其他项目的同名参数移植进来。只改本技能和必要兼容缺陷，不擅自修改推理算法。
