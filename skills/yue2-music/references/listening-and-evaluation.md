# 试听与性能证据

原生产物验收：verify_result 检验哈希；分别看 truncated.abc/semantic；FLAC 能解码、
48k stereo、有限样本、时长、RMS/peak/clipping。非零 RMS 和自然 EOS 仅证明技术
条件，不证明旋律/歌词/音乐质量。损坏、空声、截短或不完整批次不能报 pass。

生成助手 summary.json 的 wall_seconds/RTF 是 CLI 进程总时间（含加载、运行、保存，
cover 含转谱）；result timing.e2e_seconds 为 pipeline 内部时间，通常不含加载。
内存使用 sampled_maxima.physical_footprint_bytes；MLX peak 另算，不能相加。
guard 采样并不是绝对内存上限。无法读取就 null/unknown，不填 0。

compare_steps.py 测量同 semantic/noise 的 NAR+VAE 重合成；单次共享模型 load_seconds
单列，不加历史 AR 冒充新实测。已有 tools/run_fast_8step_benchmark.py 的历史结果
e2e 是旧 AR/load/transcription 加新 NAR/VAE 的估算。保留其来源，不说所有端到端
请求都更快于实时。跨模型种子相同不能保证生成相同歌曲。

要对齐官方 demo：冻结同一案例的 style/lyrics/cot/ABC 和输入哈希、源音频时长、
官方参考 URL；完整预算生成，保留失败与重试；引用官方参考和本机实测分别标注。
本机精度对比不要求与 PyTorch 随机采样生成完全一样的声音。

```bash
"$PY" -m lyra.cli listen /absolute/song-a /absolute/song-b --output /absolute/new-listening-page
```

输入必须指向真实 native song 目录（cover/song 或 helper/generate），不是父目录。
该页复制 FLAC、native 请求和 metadata；独立 summary/comparison.json 需一并交付，
不可假装内置页面已展示任意 sidecar 指标。只有用户要公开试听时再处理发布授权，
不会自动上传音频或账号信息。

人工/工具实际试听按任务评估：风格/配器、歌词可辨/遗漏重复、段落发展与 hook、
自然结尾、人声瑕疵、指定旋律/节奏遵循。若没有音频理解工具可用，提供播放器与
检查点请用户试听；不用“我听到”掩盖只做数值检查。ASR 用于辅助咬字检查，不能
替代听感；转谱误差也不能直接当生成音准错误。歌词字幕时间轴非原生产物。

最终返回绝对音频路径、试听页（如有）、模式/实际步数/种子/精度、duration/
wall time/RTF/内存及其 scope、必要的结果或问题。保留所有尝试，但突出推荐版本。
