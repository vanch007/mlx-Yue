# Historical validation status — upstream BF16 MVP

This page records the upstream hybrid M5 runtime before the local native MLX
port. Its PyTorch decoder, hardware, timings and acceptance outcomes are
historical, not claims about the current implementation. Current evidence and
remaining gates are tracked in [the local implementation](../docs/IMPLEMENTATION.md)
and [the acceptance audit](../reports/acceptance-audit/REPORT.zh-CN.md).

This release exposes the working generation pipeline with explicit open acceptance gates. It does not claim that all reference-equivalence or listening checks pass. No numerical threshold was relaxed to publish the MVP.

## Evidence included in the repository

- [mvp-results.json](mvp-results.json): curated three-song timings, process-memory and swap-out measurements, request/corpus hashes, upstream revision and retained source-report hash.
- [listening.json](listening.json): the recorded partial human assessment, scope and audio hashes. Recordings are not bundled.
- [corpus.json](corpus.json), [score.abc](score.abc) and [melody.abc](melody.abc): fixed requests and supplied-score fixtures.
- The `*-limits.json` files retain reference-derived numerical bounds and calibration provenance. AR files containing `torch211` name the corrected reference environment; earlier AR files are historical, not interchangeable acceptance limits.
- [integrity-smoke.json](integrity-smoke.json): retained rejection outcomes for invalid converted artifacts.

Large raw tensor captures, audio, environments, model weights and detailed machine logs are deliberately not included. The curated reports summarize retained local evidence, not a downloadable reproduction of every original capture. Generate new captures with the tools below; input/calibration identities must match before applying numerical limits.

## What has been exercised

- All five real-checkpoint generation paths: generated/supplied full score, generated/supplied melody score, and no score. BF16 and experimental 8-bit/4-bit AR paths have rendered; this is execution coverage, not equal quality acceptance.
- Default 32-step midpoint acoustic synthesis, retaining all 64 velocity evaluations and full attention visibility. The 64-/400-frame NAR checks pass their recorded reference-derived bounds.
- Default FP32 MPS VAE output length, halo/crop boundaries and retained waveform checks. Decoder execution was audited for unintended CPU fallback.
- Offline loading, model/conversion identities, saved plans, artifact replay, cancellation/resource guards, export and request-local reproducibility.
- Fresh BF16 conversion and verified reuse. A retained warm-filesystem local run measured 6.87 seconds for conversion and 4.99 seconds for verified reuse; these exclude downloads and are not cold-install timing promises.

### Sustained BF16 rendering

The tested machine was an M5 MacBook Air with 32 GB unified memory, on AC power. The same complete-song request was rendered three times sequentially in one process:

| Run | Audio duration | Generation wall time |
|---|---:|---:|
| 1 | 179.718667 s | 818.90 s |
| 2 | 179.718667 s | 897.01 s |
| 3 | 179.718667 s | 903.13 s |

All three naturally sampled their end tokens; neither ABC nor semantics were truncated. Sampled peak process footprint was **12.63 GiB**, with **zero additional system swap-outs**. Historical swap remained present and small swap-ins occurred: this is not a claim that swap space was empty or that no swap I/O happened. The process lifetime peak is a distinct counter and is recorded separately in [mvp-results.json](mvp-results.json).

The retained harness requires at least 180 seconds. These approximately three-minute songs **fail that strict duration gate by 0.281 seconds**. No padding, truncation or changed cutoff was used to mark the gate passed. Their full-song listening assessment remains pending.

### Human listening

The maintainer manually reviewed a **15.998667-second** reference/port pair and reported correct, perceptually identical sound. That comparison reused identical prefix, ABC, semantic IDs and initial noise, isolating NAR plus the default decoder. The semantic sequence was intentionally capped. This is not a bit-identical waveform claim and does not cover independently sampled AR output, complete-song continuity or quantized quality.

## Open acceptance gates

1. **Four strict AR comparisons remain outside current bounds:** a negative stress-cache case, long stress logits, a saved-song negative-cache case and a long valid-score-prefix cache case. The largest discrepancies were already present in prefill; the checked real-song logit argmax values agreed. This does not establish full-trajectory equivalence or an audible regression.
2. A native upstream CPU BF16 control also exceeded some MPS-derived bounds. That demonstrates that the empirical bounds are not universal across backends; it does **not** turn the MLX failures into passes. Bounds and production arithmetic remain unchanged.
3. Complete full-length matched teacher-forced AR and exact-input acoustic comparisons, plus complete-song human listening, remain pending.
4. Matched BF16, then 8-bit, then 4-bit listening is incomplete. BF16 is the conservative MVP runtime default, not the result of a completed quality-selection gate.
5. Hardware configurations beyond the 32 GB M5 Air have not been validated. No full-song, matched PyTorch/MPS-versus-MLX speedup is claimed.

## Reproduction tools

Run from the repository root after the [quick start](../README.md#quick-start). Use new output directories, AC power and one GPU workload at a time. Do not increase memory limits or shorten attention/solver work to hide a failed acceptance run.

Ordinary regression checks do not require model downloads:

```bash
uv run pytest -q
uv run ruff check src tests tools
```

The mode workload exercises bounded clips; `full` and `sustain` use full-song generation and can take many minutes. Their strict duration failure is intentionally retained:

```bash
uv run python tools/acceptance.py modes \
  --model models/converted --vae "$LYRA_VAE" \
  --precision bf16 --require-ac --output outputs/acceptance-modes
```

The reference is isolated because its dependencies differ from production:

```bash
uv venv --python 3.12 .oracle
uv pip sync --python .oracle/bin/python --require-hashes oracle/requirements.txt
PYTHONPATH=src:vendor/yue/src .oracle/bin/python tools/oracle.py --help
```

Use **PyTorch 2.11.0** from [oracle/requirements.txt](../oracle/requirements.txt) for AR reference captures. PyTorch 2.10 has an MPS two-pass BF16/fp16 attention scratch-buffer defect ([upstream fix](https://github.com/pytorch/pytorch/pull/174945)). Production keeps PyTorch 2.10 for the separate FP32 decoder, not for AR/NAR inference.

Tool responsibilities:

- [oracle.py](../tools/oracle.py): bounded reference generation, teacher-forced AR traces (including saved `--source` requests), NAR and VAE captures. Reference generation/NAR fixtures are deliberately bounded; this is not a ready-made full-song MPS benchmark.
- [fidelity.py](../tools/fidelity.py): compare MLX with saved exact-input reference tensors. Without `--limits`, it records measurements, not acceptance.
- [calibrate.py](../tools/calibrate.py) and [limits.py](../tools/limits.py): calibration evidence and bound validation. Do not reuse limits with mismatched input or reference identities.
- [acceptance.py](../tools/acceptance.py): fixed modes, full-song and sustained execution/resource reports.

Inspect each tool's `--help` for inputs and supported bounds. Full-generator FP32 MPS calibration is rejected because it is unsafe within this machine's memory budget. Equal numeric seeds in PyTorch and MLX do not guarantee equal sampled sequences; fidelity comparisons must reuse saved IDs and noise.
