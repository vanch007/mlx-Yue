# 官方完整输入重测

**旧试听报告不能作为官方效果验收证据。** 11 个 800-token 样本全部截断；旧 Cover 卡片使用了不匹配音频及写死指标。原始失败记录保留于 outputs 和 old-benchmark-audit.json。

新测试逐字读取官网原始歌词、风格与指定乐谱；完整乐谱 / 自动作曲 / off / 音频转谱链路明确区分。种子固定 2026；只因执行失败恢复，不因听感筛选重抽。断电中止的原始记录和同输入重试均保留；AR 8bit，NAR BF16，VAE FP32，32 步 midpoint，原生 9000 semantic token 上限，允许自然 EOS，不补静音、不拉伸、不人为设定最短整曲时长。

官方单曲种子、采样参数和选优预算 unknown；没有相同权重身份与随机轨迹证据，因此此处是公开输入对照，不能声称逐样本严格复现官方输出。六语种为代表性抽样，不代表覆盖 70 流派。

RTF = 端到端耗时 / 实际音频时长。端到端包含加载、校验及生成；转谱模式还包含完整音频转谱；保存另列。逐项新进程串行运行；未清除系统文件缓存，日常桌面后台仍在运行。资源在同进程每 0.1 秒采样，峰值 footprint / RSS / MLX 分列，全机 swap 增量不得自动归因于本任务。

| 案例 | 官方秒 | MLX秒 | 端到端秒 | RTF | footprint GiB | 运行 |
|---|---:|---:|---:|---:|---:|---|
| ja_score | 221.76 | 222.28 | 553.56 | 2.490 | 11.01 | pass · 运行自然结束；音乐质量 pending |
| ru_score | 119.96 | 119.96 | 195.10 | 1.626 | 10.97 | pass · 运行自然结束；音乐质量 pending |
| es_score | 182.92 | 181.04 | 389.70 | 2.153 | 11.08 | pass · 运行自然结束；音乐质量 pending |
| zh_score | 152.72 | 152.28 | 258.05 | 1.695 | 10.97 | pass · 运行自然结束；音乐质量 pending |
| ko_score | 156.92 | 156.04 | 299.78 | 1.921 | 11.15 | pass · 运行自然结束；音乐质量 pending |
| en_score | 95.76 | 95.00 | 146.93 | 1.547 | 11.16 | pass · 运行自然结束；音乐质量 pending |
| full_generated | 204.88 | 209.92 | 504.16 | 2.402 | 11.19 | pass · 运行自然结束；音乐质量 pending |
| melody_generated | 152.72 | 181.04 | 417.94 | 2.309 | 11.03 | pass · 运行自然结束；音乐质量 pending |
| melody_supplied | 107.36 | 106.96 | 154.71 | 1.446 | 10.93 | pass · 运行自然结束；音乐质量 pending |
| off_direct | 243.28 | 251.12 | 562.66 | 2.241 | 11.21 | pass · 运行自然结束；音乐质量 pending |
| cover_dagoujiao | 209.96 | 207.92 | 390.45 | 1.878 | 11.06 | pass · 运行自然结束；音乐质量 pending |
| cover_baikal | 216.92 | 212.28 | 372.66 | 1.756 | 11.15 | pass · 运行自然结束；音乐质量 pending |
| cover_birthday | 63.92 | 64.84 | 102.21 | 1.576 | 11.14 | pass · 运行自然结束；音乐质量 pending |
| audio_to_cover | 221.76 | 224.36 | 442.21 | 1.971 | 11.18 | pass · 运行自然结束；音乐质量 pending |
| jazz_funk_score | 129.96 | 130.44 | 201.32 | 1.543 | 10.96 | pass · 运行自然结束；音乐质量 pending |

## 证据与验收边界

- 15/15 项自然结束；尚不能以 EOS、时长或非零 RMS 代替歌曲完整度、歌词覆盖或音乐质量。
- 输入与来源：cases.json（完整官方记录、公开 URL、来源与输入 SHA256）；实测：measurements.json；原始请求、配置、乐谱、token、音频和资源采样：outputs/official_fullsong/<id>/。
- Genre Explorer 展示的谱是官方生成结果；本地固定此谱重合成，不等同原来从文本自动规划的流程。同一官方乐谱的六语种与 Cover 案例用于检查谱/歌词遵循。自动 Full/Melody 是相同文本的重新作曲；Melody 对带和弦官方谱属于模式消融，不能要求旋律相同。
- audio_to_cover 使用官方 scoreAudio 完整录音作为转谱源，并非旧录音替代。实际转谱 ABC 可与官方 ABC 比较，不预设转谱准确。
- 试听 MP3 是完整 FLAC 的 320 kbps 副本，无截取、归一化或拉伸；原始 FLAC 留在本机。
- 人工听评 pending。页面可记录歌词完整性、人声语言、旋律、风格、音质及结尾；未填主观分数。严格 AR/NAR 数值超限继续参考 acceptance-completion/REPORT.zh-CN.md，不因重测而豁免。
- 在线统一试听：https://vanch007.github.io/mlx-Yue/。

## 完整转谱链路的内容限制

转谱处理 225.80 秒官方乐谱录音，导出 83 小节、482 个音符（人声 0，器乐 482）。源是乐谱录音，不是真人人声演唱；本测试不代表人声旋律转谱准确率已通过。
truncated=False；warnings=[]。另有 5 项 diagnostics，不能因 warnings 为空忽略它们：
- measure 50: inferred 1/4 from downbeat span; declared numerators were [4]
- measure 59: inferred 1/4 from downbeat span; declared numerators were [4]
- measure 60: inferred 3/4 from downbeat span; declared numerators were [4, 4, 4]
- measure 72: inferred 3/4 from downbeat span; declared numerators were [4, 4, 4]
- measure 82: padded final 2/4 span to declared 4/4 with trailing rest
