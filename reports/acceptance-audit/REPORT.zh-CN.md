# YuE2 MLX 全面验收审计

结论：**不能全面验收，Request changes。** 核心生成链路已有运行证据，但存在三个已复现缺陷；原版数值对齐、完整工作流和音乐质量的验收证据尚未闭环。可以继续作为本机开发测试版本使用，不能据此交付为“全部功能已验收、质量完全等同原版”。

- 日期：2026-09-13。
- 审计版本：`8aa09f5f08bbdfd8d9a391afff0bd28d280df3fd`，`local/full-mlx`。
- 环境：Apple M3 Max / 128 GiB / macOS 27.0 / Python 3.12.10 / MLX 0.32.2。
- 范围：本地代码、测试、构建和隔离安装、已有真实模型产物、参考数值报告、五种生成路径、转谱/翻唱/编辑工具及性能证据。以项目已固定的上游版本为比较对象，不宣称涵盖未来上游变化。
- 方法：主审检查与独立转谱审查；主审重新运行缺陷复现。复现中的模拟只用于验证控制流，不冒充真实 GPU 压力或模型质量测试。
- 本次仅新增审计证据及一条项目记忆；未修复或修改生产代码。

## 已复现的必修问题

### A1 · P1 · 转谱未及时响应取消和资源保护失败

位置：[转谱入口](/Users/vanch/yue2-mlx/src/lyra/transcription/pipeline.py:71)、[转谱命令](/Users/vanch/yue2-mlx/src/lyra/music_tools/transcribe.py:11)、[翻唱入口](/Users/vanch/yue2-mlx/src/lyra/commands.py:127)、[资源检查](/Users/vanch/yue2-mlx/src/lyra/measure.py:386)。

转谱在检查 `cancelled` 前启动阻塞 FFmpeg 解码（超时上限 600 秒）并加载模型。转谱命令与翻唱入口没有保存 `GPUExecution` 对象、调用其 `check()` 或将检查接入转谱循环。后台监测已记录的内存/交换压力异常因此可能直到退出上下文才上报；期间计算和产物写入仍会继续。MLX 分配限制仍存在，问题是整进程采样保护和取消未及时传播，并非所有保护均失效。

复现：预先取消仍调用 FFmpeg；模拟运行中资源失败后，helper 先写结果，退出时才报错，过程中检查次数为零。证据：[复现脚本](/Users/vanch/yue2-mlx/reports/acceptance-audit/reproduce_transcription_cancellation.py)、[主审复现日志](/Users/vanch/yue2-mlx/reports/acceptance-audit/transcription-reproductions.log)、[独立审查说明](/Users/vanch/yue2-mlx/reports/acceptance-audit/transcription-review.md)。日志中的 PASS 表示“缺陷复现成功”，不表示被测功能通过。

验收要求：解码和模型加载前检查取消；FFmpeg 等待应可中断并回收子进程；编码/解码循环及最终写入前传播资源异常。补充预取消、运行中取消、资源失败及翻唱中止的行为测试。

### A2 · P2 · CLI 重载保存的 pipeline 会丢失生成参数

位置：[CLI 参数传递](/Users/vanch/yue2-mlx/src/lyra/cli.py:20)、[保存配置恢复](/Users/vanch/yue2-mlx/src/lyra/pipeline.py:193)。

CLI 在未指定生成配置时仍传入 `generation_config=None`。加载 `pipeline.json` 使用 `setdefault`，已有的 None 阻止恢复配置，构造器随后使用默认值。保存的步数或生成上限会被静默替换，破坏重载复现。

复现：保存 `ode_steps=17`、`semantic.max_tokens=555`，经真实 CLI 工厂和元数据加载路径后，构造器实际收到 None。证据：[脚本](/Users/vanch/yue2-mlx/reports/acceptance-audit/reproduce_cli.py)、[结果](/Users/vanch/yue2-mlx/reports/acceptance-audit/cli-reproductions.json)。模型构造被替换为参数捕获，不需要加载权重。

验收要求：明确“未提供覆盖配置”的语义并恢复已保存值；覆盖未传、显式覆盖和保存后重载三个场景。

### A3 · P2 · 合法批量任务 ID 与批次清单冲突

位置：[批量校验与写入](/Users/vanch/yue2-mlx/src/lyra/commands.py:60)。

`id="batch.json"` 通过现有校验，任务产物建立同名目录；随后程序将批次清单写到同一路径，触发 `IsADirectoryError`。错误发生在单任务异常捕获外，已经生成音频后仍会中断批次。

已通过模拟歌曲保存、真实目录操作及真实批处理逻辑复现，证据同 [CLI 复现结果](/Users/vanch/yue2-mlx/reports/acceptance-audit/cli-reproductions.json)。

验收要求：生成前拒绝保留名称，或隔离任务目录和清单位置；验证正常批次及 resume 不受影响。

## 验收矩阵

| 项目 | 状态 | 证据与边界 |
|---|---|---|
| 现有自动测试 | pass | 本次 84 passed、1 skipped、28 subtests passed；跳过的是 M5 专用路径。它们未覆盖上述三个缺陷，也不包含完整真实模型严格验收。 |
| 静态检查、sdist/wheel 构建 | pass | 本次均成功。 |
| 隔离安装与离线生成 | pass（短样本） | 新环境从 wheel 安装，从 `/tmp` 导入 site-packages；未安装/导入 Torch；禁止网络连接与域名解析的 Python audit hook 下，真实 BF16 生成成功，48 kHz、2.559 秒、64 tokens。 |
| 已有歌曲产物完整性 | pass | 27 份产物通过哈希/结构加载、FLAC 全量读取；全部数值有限、RMS 非零，有谱样本均可解析。另有本次 wheel 样本单独验证通过。 |
| 五种核心生成路径 | pass（冒烟）/ pending（完整验收） | 每条已有真实模型运行产物，但主要采用 400 tokens/约 16 秒上限，不能算自然完成歌曲。 |
| BF16 与 8bit 自然结束歌曲 | pass（运行） | 现有 4 份自然结束产物：英文固定谱与中文整曲各一对。其余 23 份主要为主动截断的测试。 |
| 保存、重载、批处理 | fail | 已有正向测试，但 A2、A3 阻断完整验收。 |
| 转谱、翻唱、编辑 | fail / pending | 原生链路有短样本证据；A1 未修复；完整翻唱、真实多窗口转谱及异常传播验证不足。编辑工具有 28 个上游 helper 子测试。 |
| 默认 MLX VAE | pass（有限范围） | 真实 40 帧参考比较 RMSE 2.82e-7、SNR 115.4 dB；长尺寸与切块的真实权重覆盖仍需补充。 |
| AR/NAR/旧 VAE/转谱参考对齐 | pending | 报告标记 measured_not_accepted，尚无与当前捕获匹配的正式通过判定。详见下节。 |
| 性能优化保持本地结果 | pass（已测样本） | 同输入 32 步完整 NAR：363.09 → 186.94 秒，1.94 倍；保留潜变量一致、解码 PCM 一致。 |
| 8bit 音乐质量与整体速度 | pending（质量）；pass（局部测速） | 固定长度三次交替测速有效；不同长度整曲不能直接比较总耗时。完整听评未闭环。 |
| 长时稳定性、其他苹果芯片 | missing evidence | 不能用历史 M5 混合后端结果代替当前 M3 原生版本；本机持续完整运行及其他硬件需分别验证。 |

测试日志：[pytest](/Users/vanch/yue2-mlx/reports/acceptance-audit-tests.log)、[lint](/Users/vanch/yue2-mlx/reports/acceptance-audit-lint.log)、[build](/Users/vanch/yue2-mlx/reports/acceptance-audit-build.log)。安装与产物证据：[wheel 验证](/Users/vanch/yue2-mlx/reports/acceptance-audit/wheel-smoke.json)、[安装日志](/Users/vanch/yue2-mlx/reports/acceptance-audit/install-online.log)、[27 份产物清单](/Users/vanch/yue2-mlx/reports/acceptance-audit/artifacts.json)。

离线安装的首次尝试因本机依赖缓存缺失失败；联网安装依赖后，断网推理通过。因此这里通过的是“依赖和模型已备齐后的离线运行”，不包含空机器离线安装。

## 数值与音乐质量尚不能签收

1. **AR**：当前 128/1024 tokens 的 logits 相对 RMS 误差约 0.464%/0.343%，argmax 一致，但 limits 为空。历史四个严格 AR 检查失败尚未在当前实现上以匹配的参考和阈值闭环；本次没有把历史失败冒充新复现，也没有将其豁免。
2. **NAR**：真实 64 帧、32 步原版对照的潜变量相对 RMS 误差约 **1.36%**，cosine 0.999907。高相关性本身不能决定是否可接受，也不能据此断言有可听退化。
3. **转谱**：真实权重对照覆盖 16 秒输入补齐到 300 秒、20 个固定解码位置，logits 相对 RMS 3.45e-6、argmax 一致；不足以验收完整解码及跨窗口拼接准确性。
4. **旧 VAE**：真实 40 帧最大绝对误差 2.57e-6，尚无新校准阈值。
5. **听评**：用户已确认音频正常，这能排除此前“空音频”的误报；不能代替原版/BF16/8bit 的整曲音质、歌词、旋律与结构遵循对比。需要相同请求的代表性完整样本与记录，而不是靠一个音频能播放判断质量等价。

参考：[AR](/Users/vanch/yue2-mlx/outputs/fidelity-ar-local/fidelity.json)、[NAR](/Users/vanch/yue2-mlx/reports/nar-real-fidelity.json)、[转谱](/Users/vanch/yue2-mlx/reports/transcription-real-parity.json)、[旧 VAE](/Users/vanch/yue2-mlx/reports/legacy_vae-real-fidelity.json)、[默认 VAE](/Users/vanch/yue2-mlx/reports/vae-real-parity.json)。不得将不同输入/参考哈希的历史 limits 套到新捕获，也不得事后放宽阈值只为通过。

性能证据：[性能报告](/Users/vanch/yue2-mlx/reports/PERFORMANCE.zh-CN.md)、[同输入整曲测量](/Users/vanch/yue2-mlx/reports/performance-summary.json)、[整曲 PCM 一致性](/Users/vanch/yue2-mlx/reports/optimized-full-audio.json)、[8bit 对比](/Users/vanch/yue2-mlx/outputs/precision-comparison/REPORT.zh-CN.md)。1.94 倍仅指该样本的声学生成阶段；短样本分批前后测试并未呈现稳定端到端提速，不能声称整套流程普遍快 1.94 倍或已达到硬件性能极限。

## 次要一致性问题

- 转谱 `warnings` 在解码出错时追加整个字典，上游追加错误字符串，可能形成混合类型列表；建议统一并覆盖异常导出场景。
- `validation/README.md` 仍以当前时态描述 M5/PyTorch MPS VAE 的历史实现，与本地原生 MLX 版本混淆；应明确标为历史证据，链接当前报告。不能把历史硬件和质量结论移用于新实现。
- `paper` 转谱预设已明确使用 FFmpeg 重采样，未复刻原版 torchaudio 论文前端；该差异不应在交付中隐藏。其他芯片未测不阻碍只针对本机的验收，但禁止扩大支持声明。

## 转为全面通过所需的最小收尾

1. 修复 A1–A3，加入对应行为回归，重跑现有测试与隔离安装检查。
2. 固定当前权重、输入、噪声和参考环境；补齐 AR 历史失败场景，按来源可追溯、输入匹配的准则完成 AR/NAR/VAE/转谱数值判定。保留不通过结果，不用局部 argmax 一致替代全链路判断。
3. 完成各声明模式的自然结束样本、一次完整翻唱，以及超过 300 秒、跨窗口的真实转谱（覆盖 full、melody-full、melody-vocal 的输出和拼接）；检查截断状态、谱面导出及边界连续性。
4. 在当前 M3 版本连续运行完整歌曲，记录内存、交换、失败恢复；按相同工作量复核主要速度结论。若仍采用现有严格时长门槛，未达标应继续标记 fail，不能通过补静音凑时长。
5. 对代表性中文/英文、不同风格与种子的完整原版/BF16/8bit 样本完成听评，记录歌词遵循、结构、异常声和衔接；量化差异与听评结论分开。统一发布文档的现行/历史状态。

这份审计已完成；功能修复及上述验收收尾尚未执行，不能据此标记项目全面完成。
