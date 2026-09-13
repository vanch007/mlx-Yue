# M5 port — architecture and provenance

> Historical design record from the original M5 port. The local `full-mlx` branch
> now uses a native MLX FP32 VAE, NumPy PCG64 acoustic noise, native
> SheetSage2/MERT2 transcription and expanded CLI workflows on M3 Max.
> See [local implementation](docs/IMPLEMENTATION.md) for current scope and evidence.
> Historical measurements and unresolved strict AR bounds below are preserved.

The BF16-first MVP implements the complete generation pipeline. Core paths and sustained full-song execution have been exercised; strict AR numerical comparisons, complete-song listening and quantized quality acceptance remain open. See the [release instructions](README.md) and [current validation status](validation/README.md).

The invariants below originated in the initial investigation and remain design constraints. The explicitly historical random-weight probes are not release benchmarks or acceptance evidence.

## Decisions

- **Port YuE2 faithfully; optimize execution, not the learned architecture.** Initial target: offline, single-song rendering on M5 Air (10 GPU cores, 32 GB). No real-time guarantee.
- **MLX Python for AR and NAR.** Adapt MLX-LM Qwen3 for AR and implement YuE2's cached acoustic path with MLX primitives, including source-specific rounding. Preserve upstream stage boundaries: `plan → generate_semantic → synthesize → decode`.
- **Upstream PyTorch/MPS is the reference, not the production AR backend.** It already has an MPS path. Start with a real checkpoint baseline rather than a separate bare-port project.
- **Keep upstream FP32 MPS VAE initially.** Proven executable and secondary in cost. No need to port every stage before accelerating the expensive ones.
- **BF16 fidelity baseline first; evaluate 8-bit AR, then 4-bit.** Keep BF16 KV and BF16 NAR initially. Quantized quality/default selection remains an empirical decision.
- Reuse fast GQA attention, RMSNorm/RoPE, quantized linears, compilation and reusable buffers. Custom Metal only for measured remaining hotspots. No Core ML/ANE-first design; M5 GPU Neural Accelerators are separate hardware and already accessible through MLX.
- Native Swift/GUI packaging is deferred. Additional Apple Silicon configurations require their own validation; the tested release target is the 32 GB M5 Air.

## Scope and provenance

- Initial feature scope: lyrics/style → song, generated or supplied ABC, upstream `full`/`melody`/`off` modes. Editing regenerates a recording; no waveform-preserving inpainting.
- Generation requires **YuE2-3B + YuE2-Vae only**. Defer SheetSage2/MERT2 transcription, source-audio covers, agent integration and best-of-N scoring. Supplied-score covers already fit generation scope.
- [Upstream source](https://github.com/multimodal-art-projection/YuE/tree/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e), package `0.1.6`; pin this commit for initial parity.
- Observed HF revisions: [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B/tree/1a96eca688d6ae5d7f0feb88573fec89920fcd19) and [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae/tree/95535e72a97bc0f09b8ada125d26b4009428c0e8). Record revisions, hashes and conversion settings in generated artifacts.
- Default listening decoder: `YuE2-Vae`. Benchmark decoder: `YuE2-Vae-legacy`; never silently substitute it.
- **Licenses:** code Apache-2.0; generator/VAE weights CC BY-NC 4.0. Do not assume commercial rights or omit upstream/third-party notices when reusing code.
- Locked production environment: Python 3.12, MLX 0.32.2, MLX-LM 0.31.3, PyTorch 2.10.0 and Transformers 5.0.0. The reference uses separate PyTorch 2.11.0 / Transformers 4.57.6 pins. MLX M5 accelerator support requires macOS ≥26.2; local measurements used macOS 26.3.
- **Subsequent oracle correction:** use the locked PyTorch 2.11.0 reference environment for real-checkpoint validation. PyTorch 2.10 MPS two-pass BF16/fp16 attention has a scratch-buffer memory-corruption defect ([fix #174945](https://github.com/pytorch/pytorch/pull/174945)). The historical random-weight measurements below are not a validated reference. See [`validation/README.md`](validation/README.md) for current environments and evidence.

## Architecture invariants and traps

### AR

- Qwen3-compatible AR subset: 28 layers, hidden 2048, FFN 6144, Q/KV heads 16/8, head dimension 128, Q/K RMSNorm, RoPE theta 1e6; vocabulary 184704, context 24576, untied embeddings.
- Use upstream `fast.py::{ar_keys,qwen_config}` as the extraction specification. Do not execute both experts per token. Approximately 2.17B AR parameters; combined learned model ≈3.58B. Checkpoint byte count also includes positional buffers.
- Preserve exact tokenization, prompt/special-token layout, sampling arithmetic/order, repetition window, minimum lengths and end-token behavior from `protocol.py` and `sampling.py`. Full/melody semantic CFG defaults to 1.0; off defaults to 1.01 and uses historical arithmetic. CFG is semantic AR only; ABC planning has none.
- Semantic output projection may select rows **`[151852:184621]`**: MUSIC_END plus 32768 codec IDs. Map local indices back to native IDs. ABC needs its own allowed output set. Preserve top-k ties/nucleus semantics when optimizing sampling.
- Keep sampling on-device; avoid full-vocabulary CPU transfers/per-token synchronizations. Bounded append-only KV storage; native GQA, not repeated KV heads. Distinguish physical cache slots from RoPE positions.

### Acoustic NAR

- Semantic tokens and 64-channel latents both run at 25 Hz. **32 midpoint steps = 64 velocity evaluations**, not 32.
- Port `nar.py::CachedNAR`, not generic `modeling_yue2.py::nar_velocity`: causal AR conditioning prefill once, retain per-layer K/V, then run only NAR experts for each velocity evaluation. NAR queries attend all visible conditioning plus bidirectional acoustic positions.
- Preserve CPU FP32 full-song noise draw/seed, BF16 solver arithmetic, timestep transform, zero boundary latents, local sinusoidal positions and global RoPE offsets.
- Original chunk capacity: `(24576 - len(prefix) - 3) // 2`. Each chunk uses the **same original prefix + local codec slice + MUSIC_END**, not accumulated preceding codec slices. Ordinary songs fit one full-song chunk.
- Query tiling must retain the complete key set. Shortening acoustic chunks/localizing attention changes model behavior; it is not equivalent memory tiling or free streaming.
- Reuse AR-generation caches only when weights, precision and prefix semantics match. **Quantized AR caches cannot replace BF16 NAR conditioning as a fidelity-preserving optimization.** Re-prefill conditioning in BF16.

### VAE and memory

- Decoder-only FP32, 48 kHz stereo; natural samples/channel = `1920*T - 64`.
- Preserve halo-and-crop decoding: upstream core 1024 frames, port default 256, halo 16; measured required halo 12. No crossfades, truncated right context or output padding. Smaller cores are a memory tradeoff, unlike smaller NAR chunks.
- Avoid unnecessary stage weight transfers/duplicate full models on unified memory. Keep large tensors resident when capacity permits; CPU artifact conversion once per stage is acceptable, not inside hot loops.
- Later VAE conversion must audit accumulation precision, including MLX TF32 behavior; an FP32 tensor dtype alone is insufficient evidence of FP32 arithmetic parity.

## Historical performance probes — not release benchmarks

Current real-checkpoint measurements are in [validation/mvp-results.json](validation/mvp-results.json). The following were short random-weight probes on battery, before checkpoint generation or thermal soak. In particular, the old long-context PyTorch 2.10 AR results are not a valid speedup baseline because of the attention defect described above.

| AR, ~6144 cached positions; no prefill/sampling | tokens/s |
|---|---:|
| Upstream MPS BF16, full output head | 8.2 |
| MLX BF16, full output head | 25.5 |
| MLX BF16, semantic-only head | 29.5 |
| MLX 8-bit, semantic-only head | 48.5 |
| MLX 4-bit, semantic-only head | 61.6 |

- At ~12K context, semantic-only BF16/8-bit/4-bit: 26.6/37.6/44.1 tok/s. KV traffic limits quantization gains.
- NAR proxy for ~215s audio, ~1K text/score prefix: Q=5378, K=11778. BF16 GEMMs 11.6–12.4 TFLOP/s; attention 52.1ms/layer/evaluation. **[INFERENCE]** ~970 TFLOP linear + ~930 TFLOP attention → ~170–180s total kernels, plus prefill/misc. Attention is comparable arithmetic to FFNs, not negligible.
- Actual upstream random-weight FP32 MPS tiled VAE: **14.05s for 215.04s audio**, correct length and finite output.
- **[INFERENCE]** Initial full-song planning ranges: ~7–10min BF16 MLX; ~5–8min quantized AR with unchanged NAR. Not measured end-to-end. AR quantization alone does not make the pipeline real-time.
- [Published 4090 reference](https://huggingface.co/m-a-p/YuE2-3B#speed-and-resources): 71.04s / 214.85s audio, 139.48 LM tok/s, 11.18 GiB peak, warm CUDA graphs/FlashAttention. Not a matched local benchmark. Quality results select 2 or 8 candidates and use legacy VAE; single-call timing excludes that selection workload.

## Deferred validation and optimization

1. Complete matched full-length teacher-forced AR and exact-input acoustic comparisons; retain the current numerical failures rather than silently changing limits.
2. Obtain complete-song human review and matched BF16, 8-bit and 4-bit listening before declaring a final quality-supported precision policy.
3. Profile and optimize the dominant acoustic synthesis stage without reducing solver steps, attention visibility or precision to claim an equivalent speedup.
4. Add upstream CLI conveniences separately: serial batch, completed-result resume, environment diagnostics and inline request flags.

Navigation under upstream `src/yue2/`: `pipeline.py` (stages/artifacts), `protocol.py` + `sampling.py` (behavior), `fast.py` (AR extraction), `modeling_yue2.py` (weights/math), `nar.py` (cached solver), `modeling_vae.py` (FP32 decoder/halo rules). Read these before implementing; preserve the source contract rather than inventing a parallel convention.
