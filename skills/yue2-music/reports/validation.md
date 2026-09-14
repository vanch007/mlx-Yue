# YuE2-music 本地验证 · 2026-09-14

已把原有项目技能升级为覆盖现有接口、需求调参和可复现产物的本地技能。
调用名 `$yue2-music`，显示名 YuE2-music；canonical 为项目 skills/yue2-music。
本地发现链接见 [local-install.json](local-install.json)。新会话若未刷新目录，可直接读取 SKILL.md。

| 检查 | 结果 / 证据 |
|---|---|
| 真实 protocol/CLI 合约 | 11 tests pass，含五模式子用例；[contract-tests.txt](contract-tests.txt) |
| 触发正例/负例/近邻规则测试 | 22/22 pass；固定语义词汇规则，非 LLM 泛化盲测；[trigger-eval.json](trigger-eval.json) |
| 六个助手 --help | 6/6 pass，在项目 .runtime 运行；[cli-help.json](cli-help.json) |
| 本机环境与权重元数据 | pass，未在 doctor 重新 hash 所有模型；[doctor.json](doctor.json) |
| 项目回归 | 107 pass、1 skip、28 subtests pass；[project-regression.json](project-regression.json) |
| 技能结构 / IR / 上下文预算 | validate_skill、resource_boundary_check pass；入口估算 960 tokens |
| 静态检查 | 新助手/合约测试及改动模块 ruff pass |
| 原生整曲 / step comparison | 见下方实际执行数据；[runtime-smoke.json](runtime-smoke.json) |

## 新助手真实执行

在 Apple M3 Max / 128 GiB / 接电 / MLX 0.32.2 / Python 3.12.10 上：
- AR 8bit，NAR BF16，FP32 默认 VAE，8 midpoint 步，默认 9000 semantic token 预算。
- 新歌曲 69.959 秒；实际 CLI 总耗时 54.680 秒，
  RTF 0.782，采样峰值 physical footprint 9.552 GiB。
- full 自动生成 ABC，两阶段自然 EOS，未截短；48 kHz stereo、FLAC、hash、非静音、
  时长一致检查 pass。该检查不包含音乐主观听评。
- 原始目录：outputs/yue2-skill-smoke-20260914；完整日志在同级 .log。

同一 semantic/noise/weights 的声学步数比较：

| 步数 | 音频秒 | NAR+VAE 实测秒 | 重合成 RTF | sampled footprint GiB |
|---:|---:|---:|---:|---:|
| 8 | 69.959 | 15.901 | 0.227 | 9.121 |
| 32 | 69.959 | 61.738 | 0.882 | 10.943 |

不包含复用的 AR 时间、共享加载和保存，不能当新整曲端到端 RTF。
重放 8 步 FLAC 与首次 8 步生成的 SHA-256 相等：True。
比较产物：outputs/yue2-skill-steps-20260914-attempt2；listen/index.html 为统一试听页。

第一轮 compare_steps 保留 stage timing 的契约错误已修复：SongResult 要求保留来源
ABC/semantic timing，不能用 0 替换。现在保留原阶段 timing，并明确 reused_stages、
source_config 和重合成计时 scope；新的 attempt2 两档均保存并校验成功。
原失败日志保留在 outputs/yue2-skill-steps-20260914.log；未覆盖失败目录。

## 必要兼容修正

doctor 和兼容 music helper 仍查询旧安装名 lyra-yue2，已改为 mlx-yue；
本机 .runtime 以离线 editable install 同步当前包元数据。README 的 cover 改用
真实 `--audio` 参数并提供含 style/lyrics 的请求文件。未修改模型算法或历史音频。

## 边界与发布级别

此次为个人本机 Scaffold；实际验证新助手及现有协议接口，未重新跑完全部历史
转谱/cover/离线导出的大模型基准。它们的技能覆盖来自当前源码和项目已有测试，
详见 [capability map](../references/capabilities.md)。
音乐主观最优、第三方盲听、任意 Apple 芯片速度保证：missing evidence，不作保证。

静态 [trust report](security_trust_report.md) 无 secret/网络脚本命中；该工具未递归
分析导入的项目代码。依赖由项目 uv.lock 管理，不复制到技能。其包装权限登记和
--help 静态识别警告保留；本机真实 --help 已另测。未制作受治理发布包，不把
会话已授权的本地文件写入/CLI 调用伪装成组织级权限签批，也不因此阻塞本地使用。
