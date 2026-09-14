# 原生生成、五模式与分阶段操作

以下命令先执行 models-and-setup 中的 `cd`、`PY`、`SKILL` 定义。
用已有模型路径，离线运行；每次输出选新目录。变量只是本地 shell 路径。

## 推荐助手：预检、实际执行

```bash
"$PY" "$SKILL/scripts/generate_music.py" --request "$SKILL/assets/prompt.json" \
  --profile standard --output "$PWD/outputs/my-song"
# 上一步只打印 effective_request；实际生成加 --execute：
"$PY" "$SKILL/scripts/generate_music.py" --request "$SKILL/assets/prompt.json" \
  --profile standard --output "$PWD/outputs/my-song" --execute
```

这是代理内部预检，不要求用户再确认已授权的生成。产物在 `my-song/generate/`；
根目录 `effective-request.json`、`decision.json`、`invocation.json`、`summary.json`
保存设置和实际耗时；失败保留 `failure.json`。助手退出码非零应读原因，截短可能
已经保存音频，不能把它标为整曲成功。只有明确诊断才用 --preview-seconds。

请求字段：style（tags 可作为别名）、lyrics、cot、seed（0 ≤ seed < 2**63）、
abc 或相对请求文件的 abc_path、cfg_scale（0–20）、filename-safe id。
顶层还有 generation_config、abc_sampling、semantic_sampling。未知参数会拒绝。

| 原生控制 | ABC 默认 | semantic 默认 |
|---|---:|---:|
| temperature | 0.7 | 1.0 |
| top_p | 0.9 | 0.95 |
| top_k | 30 | 100 |
| repetition_penalty | 1.005 | 1.2 |
| penalty_window | 100 | 50 |
| min_tokens | 32 | 200 |
| max_tokens | 4096 | 9000 |

`top_k` 必须 ≥1；temperature=0 为 greedy。generation_config 示例：
`{"ode_steps":32}`；method 固定 midpoint（32 步为 64 次 velocity evaluation），
context 固定 24576。采样 override 只改显式字段。没有 flow guidance/shift 参数。

五模式：full 无 ABC、full supplied、melody 无 ABC、melody supplied、off 无 ABC。
melody supplied 先去和弦；off 禁止 ABC。默认 full，但 cover 自动按 task 选 mode。

## 音频翻唱

```bash
"$PY" "$SKILL/scripts/generate_music.py" --request /absolute/cover-request.json \
  --action cover --audio /absolute/source.wav --task melody-full \
  --profile standard --output "$PWD/outputs/cover-song" --execute
```

cover-request 包含目标 style、真实歌词、seed，不要含 abc。输出
`cover/transcription/` 与 `cover/song/`；总耗时含转谱。目标为更可控改编时分两步：

```bash
"$PY" -m lyra.cli transcribe /absolute/source.wav --task full --offline \
  --memory-budget-gib 16 --output "$PWD/outputs/source-score"
"$PY" -m lyra.cli abc inspect outputs/source-score/score.abc
"$PY" -m lyra.cli abc strip-chords outputs/source-score/score.abc outputs/cover.abc
```

检查转谱 status/truncated、原始结果与谱，再把 cover.abc 的绝对路径放进新请求
abc_path，设 cot=melody，用 generate_music.py。需要只保留 Vocal 时，strip-chords
显式 `--keep-voice Vocal`；默认保留 Ins 旋律。翻唱不是保留原始混音或声音身份。

## 规划、编辑和复用

```bash
"$PY" "$SKILL/scripts/generate_music.py" --action plan --request /absolute/request.json \
  --output "$PWD/outputs/my-plan" --execute
# 用原生 render-plan 还原保存的配置，不能靠改命令行步数覆盖它。
"$PY" -m lyra.cli render-plan outputs/my-plan/plan --model models/converted \
  --vae /absolute/vae --precision 8bit --offline --require-ac --output outputs/rendered-plan
"$PY" -m lyra.cli replay outputs/rendered-plan --stage decode --model models/converted \
  --vae /absolute/vae --precision 8bit --offline --output outputs/redecoded
"$PY" -m lyra.cli replay outputs/rendered-plan --stage synthesize --model models/converted \
  --vae /absolute/vae --precision 8bit --offline --output outputs/resynthesized
```

plan_manifest/result 哈希校验是不可变输入合约。编辑 score.abc 要先复制到新位置，
生成时显式传该 ABC；改保存目录内部文件会校验失败。
CLI replay 保留原 ode_steps。要试新步数：

```bash
"$PY" "$SKILL/scripts/compare_steps.py" --source "$PWD/outputs/rendered-plan" \
  --steps 8 32 --output "$PWD/outputs/step-comparison"
```

该助手保留 source identity，要求完全相同的模型/VAE、token 和 noise，输出完整
native artifacts、comparison.json、listen/index.html。RTF 是 **NAR + VAE 重合成**，
不包含历史 AR 时间、共享加载和落盘。没有新端到端采样就不报告端到端加速。

## 批量与 Python

`mlx-yue batch --input requests.jsonl --concurrency 1 --output ...` 支持全部 request
及每行 generation_config。id 不分大小写唯一，不能叫 batch.json；--resume 只校验
并复用已完成记录，不是恢复中断的 token 流。失败/不完整目录改用新的 attempt。

Python：`from lyra import YuE2Pipeline, GenerationConfig, SongRequest`；
`pipe.plan → pipe.generate_semantic → pipe.synthesize → pipe.decode` 可分别调用。
`pipe(**request)` 是端到端；构造 pipeline 时给 generation_config，调用时给 sampling。
用 with 关闭权重和 GPU ownership。支持 cancelled callback、on_token、progress、
query_chunk_size、vae_core_frames、memory_budget_gib、resource_path、require_ac；
接口详细见项目 docs/usage.md 及源文件，不自创参数。
`save_pretrained` 导出离线 generator/VAE；`SongResult.save` 支持 FLAC/WAV，
`save_artifacts` 包含 PCM-24 FLAC、tokens、noise、latent、request/config/result。
