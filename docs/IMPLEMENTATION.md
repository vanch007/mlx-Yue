# Full local MLX implementation

The user approved implementing the audited missing functionality on 2026-09-13.

## Delivery sequence

1. Native FP32 MLX VAE with original checkpoint loading and matched decoder tests.
2. Torch-independent generator data/protocol/noise path; reproducibility policy recorded in each result.
3. Official CLI controls, serial batch, completed-result resume and diagnostics.
4. ABC inspection/comparison, listening artifacts and explicit MLX agent adapter.
5. Source-audio transcription and cover workflow using separately provisioned SheetSage2/MERT2, with backend and unsupported capabilities stated explicitly.
6. Real checkpoint generation, matched numerical checks and documented full-song evidence.

Pure MLX transcription is a separate model implementation from the YuE2-3B generator. Do not report a PyTorch transcription bridge as a native MLX transcription backend.

## Source references

- https://github.com/multimodal-art-projection/YuE/tree/88da114a67df892af0329472073b96a5ef700b93
- https://github.com/daig/yue2-mlx/tree/c0f0229df07daba14923627e9d78e084d82b7dc8
- https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html

## Implemented local paths

| Capability | Implementation | Evidence and limits |
|---|---|---|
| AR score planning and semantic generation | Native MLX Qwen3, BF16 and optional 8bit AR weights | Five real-checkpoint paths rendered in `outputs/native-five-modes` (intentional 400-token smoke caps); naturally ended English/Chinese recordings in `outputs/precision-comparison`; strict reference AR acceptance remains open |
| Acoustic synthesis | Native MLX BF16, original full attention and 32 midpoint steps | Saved tokens and full FP32 noise support exact-input replay; optimized full-song latents match the original local result element for element |
| Audio decoder | Native MLX FP32, original Oobleck weights and halo cropping | Real default-checkpoint 40-frame comparison: RMSE 2.82e-7, SNR 115.4 dB (`reports/vae-real-parity.json`) |
| Requests and stages | Inline/file controls, plan, render-plan, replay, cancellation | Existing API/CLI and artifact-integrity tests; resume verifies completed results and does not resume partial sampling |
| Serial batches and diagnostics | JSONL batch, upfront validation, per-request receipts, offline doctor | CLI tests cover duplicate IDs, changed requests, hash tampering and completed-result reuse |
| Score editing and listening | Official ABC inspection, chord removal, score comparison and listening-page helpers | 28 retained upstream helper subtests; score editing regenerates audio |
| Transcription and cover | Native FP32 MLX MERT2 + LoRA + BART; whole-recording window stitching; ABC/MIDI/LAB export | 16-second real transcription produced 41 notes and six measures in 4.76 s; source-audio cover rendered through the complete native chain with an intentional 1,000-token/40-second smoke cap; independent tiny Torch fixtures cover MERT and cached/full BART |
| Offline installation | Torch-free runtime and optional native transcription dependencies | `reports/native-package.json` verifies wheel contents and absence of Torch from the runtime |

The original generator's creation/cover/edit workflows are exposed locally. This
does not establish universal quality equivalence or reproduce every paper
benchmark setting. In particular, the transcription `paper` preset retains
FFmpeg resampling and explicitly records that difference from the original
torchaudio benchmark frontend. Human musical-quality evaluation remains pending.

## 8bit comparison

Only AR linear weights are quantized. Embeddings, acoustic conditioning and NAR
remain BF16; VAE remains FP32. The AR file changes from 4.332 GB to 2.656 GB.
Both variants are retained for reproducibility and acoustic conditioning.

On this M3 Max, three alternating-order, isolated-process runs per precision with
400 semantic tokens (15.999 s of audio) measured median AR speed of 59.44 token/s
for BF16 and 83.11 token/s for 8bit. Median generation time was 19.08 s versus
17.37 s. These measurements precede the new attention optimization. The short
benchmark intentionally truncates at the fixed budget and is not a quality test.

The natural-ending request pairs produce different lengths and arrangements,
so their total times are not directly comparable as speedups. Local report and
audio: `outputs/precision-comparison/REPORT.zh-CN.md` and `compare.zh.html`.
ASR is a lyric-adherence proxy, not a perceptual quality score. An empty ASR
response for the full-song 8bit sample is missing evidence, not silent audio.

## Hardware optimization policy

MLX 0.32.2 Steel attention accumulates QK, probabilities and PV in FP32 with BF16
inputs. For reviewed pre-NAX GPU generations 13–16, the local implementation
avoids redundant FP32 operand buffers. NAX and unreviewed MLX/GPU combinations
retain the original promotion. TF32 stays disabled. Query tiling, conditioning
visibility, solver steps, noise and VAE precision are unchanged.

Audited MLX source: tag `v0.32.2`, commit
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`, particularly
`mlx/backend/metal/device.cpp`, `scaled_dot_product_attention.cpp` and
`kernels/steel/attn/kernels/steel_attention.h`. Real hardware validation is on
Apple M3 Max (`applegpu_g15s`); other generations are source-reviewed only.
`tools/benchmark_nar.py` measures full 32-step solves from integrity-checked saved
tokens/noise and records conditioning-cache hashes, latent differences and time.

The earlier four strict AR failures inherited from the upstream port are not
waived. The local 128/1024-token reference captures have high agreement but no
separately calibrated acceptance limits, and remain `measured_not_accepted`.

## Latest regression evidence

- `reports/final-tests.log`: 84 passed, 1 skipped (M5-only precision path on M3), 28 upstream helper subtests passed.
- `reports/optimized-output-equivalence.json`: six BF16/8bit short runs preserve exact semantic tokens, noise, latents and decoded PCM audio.
- `reports/PERFORMANCE.zh-CN.md`: same-input full acoustic synthesis 363.09 → 186.94 seconds; 1.94× on this sample, equal conditioning caches and final latents. Short end-to-end runs did not show a speedup in the separate before/after batches; that limitation is retained in the report.
- `reports/local-validation-results.json`: command receipts for current real-workflow checks, including any failed attempts.

- `reports/legacy_vae-real-fidelity.json`: real legacy decoder, 40 identical latent frames, FP32 reference vs native MLX; maximum absolute error 2.57e-6, RMS error 2.69e-7 across full and tiled decoding. Measured without newly calibrated limits.
- `reports/transcription-real-parity.json`: real SheetSage2/MERT2 weights, same waveform and 300-second padded context; 20 fixed decoder positions and cached final position agree in argmax. Decoder-logit relative RMS error 3.45e-6; measured, not a universal transcription-quality claim.
- `reports/nar-real-fidelity.json`: real 64-frame original-Torch versus native-MLX 32-step solver; latent relative RMS error 1.36%, cosine 0.999907. This is distinct from the zero-difference before/after MLX optimization comparison; no new acceptance limits were calibrated.

- `reports/optimized-full-audio.json`: optimized 169.959-second full-song FLAC decodes to exactly the same PCM samples as the retained original.
- `reports/8bit-crop-transcription.json`: all three independently recognized 30-second excerpts of the full 8bit song contain recognizable lyrics; whole-song ASR error rate remains unscored.
