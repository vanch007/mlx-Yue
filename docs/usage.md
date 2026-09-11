# Usage and API reference

Run shell commands from the repository root after completing the [installation and model preparation](../README.md#quick-start). The shell examples assume `LYRA_VAE` was loaded from `models/paths.json` as shown there. To create an editable request, copy `examples/full-song.json` to `request.json`; requests are UTF-8 JSON. Python code belongs in a script run with `uv run python your_script.py`, or in a notebook using this project’s environment.

## CLI generation

A request is JSON. The smallest useful form is:

```json
{
  "id": "first_song",
  "style": "English, warm piano pop, expressive vocal, 88 BPM",
  "lyrics": "[Verse]\nMorning finds the open road\n\n[Chorus]\nCarry every note back home",
  "cot": "full",
  "seed": 831001
}
```

Generate from the converted model and local VAE, with all model resolution forced offline:

```bash
uv run lyra generate request.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --output outputs/first-song
```

If the portable directory was exported, it contains both model paths:

```bash
uv run lyra generate request.json --model models/offline --offline --output outputs/first-song
```

Output directories must be absent or empty; Lyra never mixes or silently overwrites recordings. `--quiet` disables display progress without changing RNG or generation. `--require-ac` rejects a run that starts off AC power or loses AC power. CLI runs also write sampled resource evidence beside the output as `<output>.resources.jsonl` and `<output>.resources.json`.

A supplied score is read as UTF-8 while preserving its bytes through tokenization. `--abc` and `--mode` override the corresponding request fields:

```bash
uv run lyra generate request.json \
  --abc validation/score.abc \
  --mode full \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --offline \
  --output outputs/supplied-score
```

The five valid paths are:

| `cot` | ABC | Behavior |
|---|---|---|
| `full` | generated | Generate melody and chord ABC, then audio |
| `full` | supplied | Use exact supplied melody/chord ABC |
| `melody` | generated | Generate melody-plan ABC; accompaniment remains free |
| `melody` | supplied | Use exact supplied melody ABC |
| `off` | none | Generate semantic music directly; supplied ABC is invalid |

## Requests and generation controls

`SongRequest` fields are `style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale`, and filename-safe `id`. Python also accepts `tags` as an alias for `style`, but the two cannot disagree. Seeds are integers in `[0, 2**63)` and belong to one request.

A CLI request may contain `generation_config`, `abc_sampling`, and `semantic_sampling`. In Python, pass `GenerationConfig` when constructing the pipeline and pass stage-local sampling overrides to `plan`, `generate_semantic`, or the end-to-end call. A sampling override dictionary changes only named fields.

| Control | ABC default | Semantic default |
|---|---:|---:|
| `temperature` | 0.7 | 1.0 |
| `top_p` | 0.9 | 0.95 |
| `top_k` | 30 | 100 |
| `repetition_penalty` | 1.005 | 1.2 |
| `penalty_window` | 100 | 50 |
| `min_tokens` | 32 | 200 |
| `max_tokens` | 4096 | 9000 |

`temperature=0` selects greedy decoding. The semantic CFG default is `1.0` for `full`/`melody` and the upstream historical `1.01` arithmetic for `off`; `cfg_scale` explicitly overrides it in `[0, 20]`. ABC planning has no CFG. The acoustic default is 32 midpoint steps, which means 64 velocity evaluations. `ode_steps` is configurable, while `ode_method="midpoint"` and context `24576` are fixed protocol fields. Prefix plus requested generation budget overflow fails explicitly. Exhausting `max_tokens` sets the corresponding truncation flag rather than pretending an end token was sampled.

Example fields to merge into a complete request JSON:

```json
{
  "generation_config": {"ode_steps": 32},
  "abc_sampling": {"top_k": 30},
  "semantic_sampling": {"top_k": 80, "top_p": 0.95},
  "cfg_scale": 1.2
}
```

## Python API and independent stages

The end-to-end call returns a `SongResult`:

```python
import os

from lyra import YuE2Pipeline

with YuE2Pipeline.from_pretrained(
    "models/converted",
    vae=os.environ["LYRA_VAE"],
    local_files_only=True,
    precision="bf16",
    memory_budget_gib=16,
    vae_core_frames=256,
    query_chunk_size=256,
    progress=False,
) as pipe:
    song = pipe(
        style="English, warm piano pop, expressive vocal, 88 BPM",
        lyrics="[Verse]\nMorning finds the open road",
        cot="full",
        seed=831001,
        semantic_sampling={"top_k": 80},
    )
    song.save("outputs/first-song.wav")
    song.save_artifacts("outputs/first-song-artifacts")

print(song.truncated)
```

Use `.flac` or `.wav`; FLAC is written as PCM-24 and WAV as float audio. `save_artifacts` always includes FLAC plus the replay inputs described below.
Set `resource_path="outputs/python.resources.jsonl"` to retain sampled guard data and its companion `.json` report from a Python pipeline; set `require_ac=True` when AC continuity is required.

The same pipeline exposes each stage directly:

```python
import os

from lyra import GenerationConfig, SongRequest, YuE2Pipeline

request = SongRequest(
    style="English, warm piano pop, expressive vocal, 88 BPM",
    lyrics="[Verse]\nMorning finds the open road",
    cot="full",
    seed=831001,
    id="staged_song",
)

with YuE2Pipeline.from_pretrained(
    "models/converted",
    vae=os.environ["LYRA_VAE"],
    local_files_only=True,
    generation_config=GenerationConfig(ode_steps=32),
) as pipe:
    plan = pipe.plan(request=request, abc_sampling={"top_k": 30})
    semantic = pipe.generate_semantic(plan, sampling={"top_k": 80})
    latents = pipe.synthesize(semantic)
    audio = pipe.decode(latents)
```

- `SymbolicPlan` retains the request, ABC text, original ABC token IDs, exact semantic prefix, timing, and ABC truncation state.
- `SemanticResult` retains the plan, codec-local semantic IDs, timing, and semantic truncation state.
- `synthesize` returns finite CPU FP32 `[T, 64]` latents. An optional finite CPU FP32 `noise=` array can replay an exact solver input.
- `decode` accepts `[T, 64]` or `[1, 64, T]` latents and returns finite CPU FP32 `[samples, 2]` audio.
- `SongResult` exposes `audio`, `sample_rate`, `abc`, `semantic`, `latents`, retained `noise`, effective `config`, weight identities, stage/load timing, request identity, and separate ABC/semantic truncation flags.

## Exact plans, editing, and artifact replay

Save a plan without rendering it:

```bash
uv run lyra plan request.json \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/plan
```

Render the exact saved plan later. The saved generation configuration and original ABC IDs/prefix are restored directly; ABC is not decoded and re-tokenized:

```bash
uv run lyra render-plan outputs/plan \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/from-saved-plan
```

A saved plan is immutable integrity-checked input. Do not edit `outputs/plan/score.abc` in place: its manifest will reject the change. To edit a composition, copy the score to a new file, edit it, and submit that file as a new external plan input:

```bash
cp outputs/plan/score.abc edited.abc
# Edit edited.abc, then generate a new recording:
uv run lyra generate request.json \
  --abc edited.abc --mode full \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/edited-recording
```

Edit `request.json` as well when changing lyrics or style. Every such edit intentionally creates a new request and recording.

A full artifact directory can restart at either retained latents or retained semantic tokens:

```bash
# Reuse latents; run only the decoder.
uv run lyra replay outputs/first-song-artifacts --stage decode \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/redecoded

# Reuse semantic tokens and solver noise; rerun NAR and the decoder.
uv run lyra replay outputs/first-song-artifacts --stage synthesize \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/resynthesized
```

If an older artifact lacks retained noise, `--stage synthesize` regenerates the original request-local CPU noise from the saved seed and records that fact. Replay records the source identity, source weights/configuration/artifact hashes, truncation, and whether latents or noise were reused.

### Artifact contents and provenance

A saved plan contains `plan.json`, `plan_manifest.json`, `abc_tokens.npy`, `prefix.npy`, and `score.abc` when applicable. A saved song adds `request.json`, `config.json`, `semantic.npy`, `latent.npy`, optional retained `noise.npy`, `audio.flac`, and `result.json`.

`result.json` hashes every other artifact and records request/configuration/weight identity, stage timings, sample rate, duration, and separate truncation flags. `config.json` records effective sampling/CFG, backend and dtypes, geometry, RNG policy, runtime versions/hash, source commit, and decoder release. `weights` includes the complete conversion manifest for the generator and the verified VAE identity. A `status` of `complete` means the requested operation finished; it does not override `truncated.abc` or `truncated.semantic`.

## Cancellation, callbacks, and ownership

Python calls accept:

```python
song = pipe(
    **request,
    cancelled=stop_event.is_set,
    on_token=lambda phase, native_id: record(phase, native_id),
)
```

`cancelled` is polled before and during AR prefill/generation, before and during acoustic prefill/flow matching, and before VAE decoding. Cancellation raises `InterruptedError`. The token callback receives each actual native-ID ABC or semantic output once, including its end token; prefixes, supplied ABC, and CFG-only branch work are not reported as outputs. Display progress can be disabled independently.

Per-request AR caches and per-chunk NAR state close in `finally` paths, so a cancelled request does not become implicit input to the next serial request. A resource-limit failure is latched by the execution guard and deliberately keeps failing that guarded pipeline. Use a context manager (or call `close`) to release resident model references, clear MLX/MPS caches, and release ownership. Results already copied to CPU remain usable after closing.

One pipeline may retain weights between serial requests, but concurrent calls on one instance are unsupported. A process/thread lock allows only one Lyra GPU workload at a time; another process fails rather than contending for unified memory. Allocator limits are process-global and remain conservatively configured after the context exits.

## Faithful execution and memory geometry

- **AR:** MLX executes only the required Qwen3-compatible expert path with the pinned tokenizer, exact prompt/special-token layout, phase-local native-ID output projections, upstream minimum/end/repetition/top-k/top-p/CFG behavior, device-local sampling, absolute cache/RoPE positions, and 1024-token prefill chunks. BF16 mode keeps weights and caches in BF16. Each AR phase creates its own request-local MLX key from the full 63-bit seed and never mutates global MLX RNG state.
- **NAR:** MLX preserves BF16 conditioning and solver arithmetic, two zero boundary latents, local learned latent positions, globally offset RoPE, the source timestep shift, and midpoint updates. The original chunk capacity is `min((24576 - len(prefix) - 3) // 2, 24576)` semantic frames. Each chunk prefills causal conditioning exactly once, then uses the original prefix plus only its local codec slice and `MUSIC_END`; earlier codec chunks are not accumulated. The CPU FP32 noise tensor is drawn once for the whole song with a request-local Torch generator, then sliced at those original cuts.
- **Attention:** the default NAR query tile is 256 positions, but every query still sees the full conditioning and acoustic key set. Query tiling is memory geometry, not local attention or streaming. Full attention uses temporary FP32 operands with MLX TF32 disabled, then rounds output to BF16; the short unmasked vector kernel already has FP32 opmath. Retained weights, KV and solver state stay BF16. Quantized AR generation caches are discarded before BF16 NAR conditioning.
- **VAE:** the default `YuE2-Vae` decoder is the original decoder-only FP32 PyTorch implementation on MPS. Default tiling is 256 latent-frame cores with a 16-frame halo and exact crop; there are no crossfades, shortened right context, or output padding. For `T` latent frames, output is 48 kHz stereo with `1920*T - 64` samples per channel. An alternate decoder must be passed explicitly; the benchmark `YuE2-Vae-legacy` is never silently substituted.

The default guard is a **sampled 16 GiB whole-process budget**, not a promise that framework counters sum to process use. At that budget it configures an 11 GiB **advisory** MLX limit, a 128 MiB MLX cache limit, and a 15 GiB **hard combined-Metal limit enforced when MPS allocates**. PyTorch's MPS allocator counts other Metal allocations, including MLX buffers; this is not a separate decoder allowance. Its driver-memory counter likewise overlaps MLX usage.

The monitor samples physical footprint, system availability/pressure, swap I/O, MLX counters, and MPS counters every 0.25 seconds. It stops at a sampled footprint over budget, non-normal pressure, less than 2 GiB available, more than 64 MiB new swap-outs, or more than 128 MiB swap-used growth from entry. Existing swap at entry is not a failure. Because process, headroom, pressure, and swap checks are sampled, bounded prefill/query/VAE tiling remains required.

## Reproducibility contract

Within the supported MLX runtime and unchanged request/configuration/weights, AR sampling and acoustic noise are request-local and reproducible. ABC and semantic phases each reset their own MLX stream from the request seed, matching the source phase policy; the acoustic stage independently uses the same seed for one full-song CPU FP32 noise draw. Equal numeric seeds across PyTorch and MLX do **not** promise identical sampled songs because their categorical RNG implementations differ. Fidelity comparisons therefore reuse identical saved IDs and noise rather than comparing independently sampled songs.


## Portable offline models

After preparation, export a self-contained generator and decoder directory (this makes another copy of the weights):

```python
import os
from lyra import YuE2Pipeline

with YuE2Pipeline.from_pretrained(
    "models/converted", vae=os.environ["LYRA_VAE"], local_files_only=True,
) as pipe:
    pipe.save_pretrained("models/offline")
```

The resulting directory can be used with `--model models/offline --offline`, without `--vae`. Keep the locked Python environment as well as the model directory.

## Experimental AR quantization

BF16 is the MVP default. The optional variants below have not completed the matched listening assessment; do not treat them as equally validated defaults.

```bash
uv run lyra prepare --cache-dir models/hf-cache --output models/converted --precision 8bit --offline
uv run lyra prepare --cache-dir models/hf-cache --output models/converted --precision 4bit --offline
```

These use affine, group-size-64, linear-only AR quantization. Both preserve the BF16 partitions required for acoustic conditioning and synthesis; they add files rather than quartering the complete installation’s disk size. Pass the matching `--precision` to generation. Cached pinned source snapshots are required when preparing variants.

See [validation status](../validation/README.md) and [architecture/provenance](../PORTING.md) for the current limitations and source identities.
