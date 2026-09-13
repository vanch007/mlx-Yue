# yue2-mlx

Run [YuE2](https://huggingface.co/m-a-p/YuE2-3B) music generation locally on Apple Silicon. This local `full-mlx` branch runs autoregressive planning, acoustic synthesis and the FP32 VAE in MLX. Optional SheetSage2/MERT2 transcription also runs in MLX. After model preparation, generation works offline.

**Local implementation under acceptance testing.** Generation, native transcription, serial batches, verified completed-result resume, score tools and listening pages are implemented. Strict long-context AR reference comparisons and human listening remain separate acceptance items. The project is independent of the upstream YuE team. See [local implementation and evidence](docs/IMPLEMENTATION.md); historical upstream-port measurements below retain their original scope.

The Python distribution is `lyra-yue2`; commands are **`yue2-mlx`** and **`lyra`**, with Python import `lyra`.

## Requirements

- **Local native-MLX testing:** M3 Max, **128 GB unified memory**, macOS 27.0. Historical hybrid-runtime measurements used an M5 MacBook Air with 32 GB. Intel Macs, Linux and Windows are not supported by this runtime.
- **macOS 26.2 or newer**, Python **3.12**, and [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv` can provision the required Python version.
- Internet access for initial dependencies and approximately **7.8 GB of model downloads**. Model weights are downloaded from their pinned upstream Hugging Face repositories, not from this GitHub repository.
- **At least 20 GB of free disk space** for a BF16 setup. The original generator, converted copy and decoder occupy approximately **15.1 GB**, before Python dependencies, caches and generated recordings. Larger margins are useful for experiments.
- Connect AC power, close other memory-heavy workloads, and run only one generation at a time. The runtime enforces a sampled **16 GiB process budget** and stops on unsafe memory pressure or new swapping.

Use the dedicated environment created below. The package includes a vendored top-level `yue2` module and must not be co-installed with upstream `yue2-infer` in the same environment.

## Quick start

### 1. Install

```bash

# Run from this local checkout. The full-mlx changes are local.
uv sync --frozen --no-dev --extra transcription
source .venv/bin/activate
export MLX_ENABLE_TF32=0
```

PyTorch is needed only for development/reference tests. Use `uv run --no-dev --extra transcription` for a Torch-free runtime, or invoke `.venv/bin/yue2-mlx` directly after installing. Transcription additionally requires `ffmpeg`. If MLX was already initialized with reduced FP32 precision in a notebook, restart its kernel before importing `lyra`.

### 2. Download and convert once

This is fully automatic: it downloads the pinned generator and default decoder, checks source identities and tensor layouts, and losslessly repackages BF16 generator tensors for this runtime. No manual checkpoint editing, training or calibration is needed.

```bash
mkdir -p models &&
yue2-mlx prepare \
  --cache-dir models/hf-cache \
  --output models/converted \
  --precision bf16 > models/paths.json &&
LYRA_VAE="$(python -c 'import json; print(json.load(open("models/paths.json"))["vae"])')" &&
export LYRA_VAE
```

The returned model paths are saved in `models/paths.json`; the decoder path is captured automatically. Continue only after this command succeeds. Network, permissions or disk-space errors must be resolved before generation. Re-running preparation verifies and reuses a valid conversion; incomplete staged weights are not installed as a completed model.

Keep `models/converted` and `models/hf-cache` in place. In a **new terminal**, return to this repository and restore the decoder path:

```bash
LYRA_VAE="$(python -c 'import json; print(json.load(open("models/paths.json"))["vae"])')" &&
export LYRA_VAE
export MLX_ENABLE_TF32=0
```

### 3. Generate a short first clip

```bash
yue2-mlx generate examples/quickstart.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --require-ac \
  --output outputs/quickstart
```

The included English piano-pop request uses a supplied score and a **400-token semantic budget**, producing at most approximately **16 seconds** of audio. It intentionally permits truncation for a short installation check; it is not a complete-song quality example. It retains the normal **32 midpoint solver steps**.

The recording is `outputs/quickstart/audio.flac`. `result.json` records timing, identities and truncation flags; intermediate artifacts allow later replay. On macOS, listen with:

```bash
open outputs/quickstart/audio.flac
```

Output directories must be absent or empty. Use a new name such as `outputs/quickstart-2` to render again; recordings are never silently overwritten. `--quiet` disables progress display.

### 4. Generate a full song

```bash
yue2-mlx generate examples/full-song.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --require-ac \
  --output outputs/full-song
```

This is the Chinese City Pop example from the [retained acceptance corpus](validation/corpus.json), attributed there to the upstream demo. It generates its own score and uses the normal generation budgets. Copy the JSON file and edit `style` and `lyrics` for your own request. Song duration is generated, not a fixed-length promise; inspect the truncation flags when a token budget is reached.

**Measured on the 32 GB M5 Air:** three sequential BF16 runs of this request produced naturally ended **179.719-second** songs in **13.6–15.1 minutes each**, with **12.63 GiB sampled peak process footprint** and no additional swap-outs. Timings vary with the request and machine. These songs miss the unchanged strict 180-second harness cutoff by 0.281 seconds; they were not padded. Full-song listening remains pending. See the [public measurement summary](validation/mvp-results.json) and [validation status](validation/README.md).

## Python API

Local M3 Max performance evidence: the same 169.96-second song's acoustic stage
ran in **363.09 s before** and **186.94 s after** the reviewed Steel-attention
optimization, with identical conditioning caches and final latents (32 steps,
16 GiB budget). This is a single full-song pair, not a universal speed claim.
Short end-to-end runs did not demonstrate a stable improvement. See the detailed
[local performance report](reports/PERFORMANCE.zh-CN.md) and its numerical evidence.

Run scripts with `python your_script.py` from this checkout:

```python
import json
from pathlib import Path
from lyra import YuE2Pipeline

paths = json.loads(Path("models/paths.json").read_text())
request = json.loads(Path("examples/quickstart.json").read_text())

with YuE2Pipeline.from_pretrained(
    paths["model"],
    vae=paths["vae"],
    precision="bf16",
    local_files_only=True,
    require_ac=True,
) as pipe:
    song = pipe(**request)
    song.save("outputs/python-clip.wav")
    print(song.truncated)
```

The API also exposes `plan → generate_semantic → synthesize → decode`, saved plans, exact-input acoustic replay, cancellation, token callbacks and WAV/FLAC export. See the [usage and API reference](docs/usage.md) for controls, score editing, artifacts and portable offline model directories.

## Local workflow additions

Use a fresh output directory for every new recording. `--resume` verifies a completed recording's request, settings, weights and artifact hashes; it does not resume a partly sampled song.

```bash
yue2-mlx generate --style 'warm piano pop' --lyrics-file lyrics.txt --seed 42 \
  --model models/converted --vae "$LYRA_VAE" --offline --output outputs/song
yue2-mlx batch --input requests.jsonl --model models/converted --vae "$LYRA_VAE" \
  --offline --output outputs/batch --resume
yue2-mlx doctor --model models/converted --vae "$LYRA_VAE" --verify-hashes
yue2-mlx transcribe recording.wav --task melody-full --output outputs/score
yue2-mlx cover request.json --audio recording.wav --task melody-full \
  --model models/converted --vae "$LYRA_VAE" --output outputs/cover
yue2-mlx abc inspect outputs/score/score.abc
yue2-mlx listen outputs/song outputs/cover/song --output outputs/listen
```

`transcribe` supports full harmony+melody, full melody, and vocal melody. It downloads separately pinned SheetSage2/MERT2 checkpoints on first use; add `--offline` after preparation. Default whole-song inference uses the official 300-second encoder context with overlap, including silence padding for short recordings. Cover keeps the transcript and generation artifacts in separate subdirectories. Existing ABC can also be supplied to `generate --abc-file score.abc --mode melody` or edited and rendered as a new request.

8bit conversion adds an AR weight variant while retaining BF16 acoustic conditioning and FP32 VAE:

```bash
yue2-mlx prepare --offline --cache-dir models/hf-cache --output models/converted --precision 8bit
yue2-mlx generate request.json --model models/converted --vae "$LYRA_VAE" \
  --precision 8bit --offline --output outputs/song-8bit
```

## Acceptance boundaries

- Historical upstream-port evidence covers five generated/supplied-score paths across `full`, `melody` and `off`, and a manually reviewed 16-second reference/port pair. These were measured before this local native VAE/transcription implementation and do not substitute for its acceptance results.
- Local evidence is tracked in [implementation status](docs/IMPLEMENTATION.md). The default decoder is `YuE2-Vae`, not the legacy benchmark decoder.
- Four historical strict AR comparisons remain outside their empirical bounds and have not been waived. Local measured comparisons without calibrated limits are explicitly marked as such; complete-song perceptual equivalence remains open.
- **BF16 is the MVP default.** Optional 8-bit/4-bit AR modes remain experimental pending listening. They retain BF16 acoustic conditioning and add weight files rather than shrinking the whole installation to one quarter.
- Local batch, resume, doctor and inline request controls are implemented and covered by tests.
- Editing a score, style or lyrics regenerates audio; it is not waveform-preserving inpainting. Streaming audio, real-time guarantees, benchmark best-of-N ranking and a native Mac GUI are not part of the official generation API or this local implementation.

For diagnostics, include the command, macOS/chip/memory information, exception and relevant stage timings in a [GitHub issue](https://github.com/daig/yue2-mlx/issues). Redact private lyrics, local paths and credentials before sharing reports.

## Development and provenance

```bash
uv run pytest -q
uv run ruff check src tests tools
```

Model-based acceptance and the separately locked PyTorch reference are described in [validation](validation/README.md). Architecture invariants and pinned source/model revisions are in [PORTING.md](PORTING.md). Raw tensor captures, research outputs, environments and model weights are deliberately excluded from the repository and release packages.

## Licenses

- Project code: [Apache-2.0](LICENSE). Vendored upstream code retains its [Apache license](vendor/yue/LICENSE).
- **Model weights: [CC BY-NC 4.0](vendor/yue/MODEL_LICENSE)**, including the generator and default decoder. The code license does not grant unrestricted commercial model use.
- VAE-derived code retains the [upstream third-party notices](vendor/yue/THIRD_PARTY_NOTICES.md) and [MIT license texts](vendor/yue/licenses/). The adapted MLX-LM attention retains [Apple's MIT license](src/lyra/licenses/MLX_LM_MIT.txt).
- The optional SheetSage2 transcription utilities retain their [source attribution](src/lyra/transcription/upstream/SOURCE.json) and upstream repository license scope; the Apache project license does not relicense them or the transcription weights.

No model weights are bundled or re-hosted by this release.
