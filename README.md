# mlx-Yue

Native Apple Silicon (MLX) port for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) music generation and audio transcription.

[![GitHub](https://img.shields.io/badge/GitHub-vanch007%2Fmlx--Yue-blue?logo=github)](https://github.com/vanch007/mlx-Yue)
[![Demo](https://img.shields.io/badge/%F0%9F%8E%A7%20Demo-Audio%20Showcase-orange)](https://vanch007.github.io/mlx-Yue/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-vanch007%2Fmlx--Yue2--3B-yellow)](https://huggingface.co/vanch007/mlx-Yue2-3B)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey?logo=apple)](https://github.com/vanch007/mlx-Yue)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**mlx-Yue** brings the state-of-the-art YuE2-3B music foundation model natively to Apple Silicon via Apple's [MLX](https://github.com/ml-explore/mlx) framework. It features an end-to-end Torch-free inference pipeline, optimized Metal SDPA kernels, 8-bit AR quantization for faster generation, and the full suite of creation, transcription, and cover workflows.

Pre-converted model weights (BF16 & 8-bit quantized) are available on Hugging Face:
👉 **[vanch007/mlx-Yue2-3B](https://huggingface.co/vanch007/mlx-Yue2-3B)**

Interactive Audio Showcase (Official Demos vs 32-Step Standard vs 8-Step Fast Mode):
👉 **[https://vanch007.github.io/mlx-Yue/](https://vanch007.github.io/mlx-Yue/)**

Featuring 15 complete songs evaluated across 6 languages and 5 generation modes. Includes side-by-side listening comparison between official demo references, 32-step standard mode, and 8-step ultra-fast mode (<1.0 RTF, faster-than-realtime generation on Apple Silicon). See the full [Benchmark Report](reports/official-fullsong/REPORT.zh-CN.md).

---

## Key Highlights

- 🍎 **Native Apple Silicon Architecture**: Autoregressive (AR) planning, acoustic flow matching (NAR), and high-fidelity FP32 Oobleck VAE decoder all execute directly in MLX without PyTorch runtime dependencies.
- ⚡ **8-Bit AR Quantization**: Supports 8-bit quantization on AR linear layers, reducing memory footprint by ~40% (4.33 GB $\to$ 2.66 GB) and boosting AR token generation from ~59 tokens/s to ~83 tokens/s on M3 Max.
- 🎼 **All Five Generation Modes Supported**:
  1. **Full + Generated Score**: Fully automated song writing, score planning, and 48kHz audio generation from text prompt & lyrics.
  2. **Full + Supplied ABC Score**: Compose songs conditioned on user-supplied ABC notation.
  3. **Melody + Generated Score**: Automatic lead-sheet generation and vocal/melody arrangement.
  4. **Melody + Supplied ABC Score**: Condition acoustic synthesis on an exact melody line.
  5. **Off (Direct Generation)**: Generate music from style and lyrics without a symbolic score. This mode can include vocals; it is not an instrumental-only switch.
- 🎙️ **Full Audio Transcription & Cover Chain**:
  - Native SheetSage2 + MERT2 music transcription in MLX (Audio $\to$ ABC notation / MIDI / LAB).
  - Multi-window stitching supporting arbitrarily long source tracks.
  - End-to-end **Cover** pipeline: Transcribe source audio $\to$ extract melody/structure $\to$ re-synthesize with new styles.
- 🚀 **Metal Performance Shaders (MPS / Steel) Optimization**:
  - Leverages MLX Steel SDPA with FP32 accumulator on Apple Silicon to maximize memory bandwidth and GPU compute utilization.
  - Same-input full acoustic synthesis accelerated from **363.09 s to 186.94 s** (1.94× speedup) with bit-identical latent representations.
- 🛡️ **Intelligent Resource Guard**: Real-time memory watchdog enforcing a sampled 16 GiB process budget to prevent swap thrashing and out-of-memory crashes on Apple Silicon unified memory.

---

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4/M5 series). Tested on M3 Max (128 GB) & M5 Air (32 GB).
- **macOS 14.2+** for M1–M4; **macOS 26.2+** for M5. Use native arm64 Python, not Rosetta. The macOS 15.7.7 / M3 Ultra platform gate has regression coverage; physical tests by this project used M3 Max and M5 Air.
- **Python 3.12** and [uv](https://docs.astral.sh/uv/) package manager.
- `ffmpeg` (required if running audio transcription / cover features).

---

## Quick Start

### 1. Installation

Clone this repository and create the isolated MLX environment:

```bash
git clone https://github.com/vanch007/mlx-Yue.git
cd mlx-Yue

# Install dependencies (frozen lockfile, Torch-free native runtime)
uv sync --frozen --no-dev
source .venv/bin/activate
export MLX_ENABLE_TF32=0
```

For transcription and cover, add the optional audio tools:

```bash
uv sync --frozen --no-dev --extra transcription
```

PyTorch is only used by development/reference tests. Generation and transcription
installations above are Torch-free. Avoid plain `uv sync` for production: it also
installs the development group. After updating the checkout, repeat the same sync
command and run `mlx-yue doctor`.

Dependency pins and platform fixes responding to the M3 Ultra community benchmark
are documented in [the verification report](reports/x-feedback-20260915/REPORT.md).
To repeat the known-vulnerability check (network access required):

```bash
python tools/audit_dependencies.py --profile runtime --output outputs/dependency-audit
# Other profiles: transcription, dev (includes transcription and test dependencies)
```

This checks the selected locked dependency graph on your platform against the
current PyPI advisory database. It is not a malware scan or a guarantee against
unknown vulnerabilities. Historical `oracle/requirements.txt` pins preserve old
numerical experiments and are not the recommended installation environment.

---

### 2. Model Weights

#### Option A: Download Pre-converted Weights from Hugging Face (Recommended)

Pre-converted weights are hosted at **[vanch007/mlx-Yue2-3B](https://huggingface.co/vanch007/mlx-Yue2-3B)**:

```bash
# Download model weights to models/converted
python -c '
from huggingface_hub import snapshot_download
snapshot_download(repo_id="vanch007/mlx-Yue2-3B", local_dir="models/converted")
'
```

The pre-converted repository contains:
- `ar-bf16.safetensors` & `ar-8bit.safetensors` (Autoregressive models)
- `nar-bf16.safetensors` (Acoustic Flow-Matching NAR model)
- `qwen.tiktoken` & model configs

For the VAE decoder, download from [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae):
```bash
python -c '
from huggingface_hub import snapshot_download
snapshot_download(repo_id="m-a-p/YuE2-Vae", local_dir="models/vae")
'
export LYRA_VAE="$(pwd)/models/vae"
```

#### Option B: Automatic Conversion from Official Upstream

Alternatively, convert weights automatically from the official Hugging Face repository:

```bash
mlx-yue prepare \
  --cache-dir models/hf-cache \
  --output models/converted \
  --precision bf16 > models/paths.json

LYRA_VAE="$(python -c 'import json; print(json.load(open("models/paths.json"))["vae"])')"
export LYRA_VAE
```

To convert an **8-bit quantized AR model**:
```bash
mlx-yue prepare \
  --cache-dir models/hf-cache \
  --output models/converted \
  --precision 8bit
```

---

### 3. Command-Line Usage

#### A. Generate a Short Clip (Quickstart Verification)
```bash
mlx-yue generate examples/quickstart.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --require-ac \
  --output outputs/quickstart
```

#### B. Generate a Full Song
```bash
mlx-yue generate examples/full-song.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision 8bit \
  --offline \
  --require-ac \
  --output outputs/full-song
```

Listen to the generated 48kHz lossless FLAC:
```bash
open outputs/full-song/audio.flac
```

#### C. Music Transcription (Audio to Sheet / MIDI)
```bash
mlx-yue transcribe input_song.wav \
  --output outputs/transcription \
  --task melody-full
```

#### D. End-to-End Cover (Style Transfer / Re-arrangement)
```bash
mlx-yue cover examples/full-song.json --audio input_song.wav \
  --style "r&b, soulful female vocal, warm rhodes piano" \
  --output outputs/cover
```

---

## Local agent skill

[YuE2-music](skills/yue2-music/SKILL.md) covers all five generation modes, transcription,
audio covers, score editing, saved plans, replay and serial batches. It selects settings
from the musical brief and provides 8-step fast, 32-step standard and BF16 reference
profiles, with full-song budgets, verified artifacts and measured timing.

The [skill guide](skills/yue2-music/references/generation-and-covers.md) includes a
request validator/generator and a controlled 8/32-step listening comparison helper.
Invoke it in a skill-enabled local agent with `$yue2-music`, followed by your brief.
Keep the project skill as the canonical source when linking it into your agent's skill directory.

## Python API

You can easily integrate `mlx-Yue` into Python projects:

```python
import json
from pathlib import Path
from lyra import YuE2Pipeline

# Load pre-trained MLX pipeline
pipe = YuE2Pipeline.from_pretrained(
    "models/converted",
    vae="models/vae",
    precision="8bit",        # Choose '8bit' or 'bf16'
    local_files_only=True,
    require_ac=True,
)

# Define song creation request
request = {
    "style": "city pop, 80s japanese groovy bass, bright brass, female vocal",
    "lyrics": """[verse]
Neon lights reflect in the rainy avenue
Whispering shadows dancing through the midnight blue
[chorus]
Stay with me tonight under the city glow
Let the rhythm guide us where the night winds blow
""",
    "seed": 42
}

# Generate song
result = pipe(**request)

# Save high-definition FLAC audio and notation artifacts
result.save_artifacts("outputs/my_song")
print(f"Generated {result.audio_seconds:.1f}s song at outputs/my_song/audio.flac")
```

---

## Performance & Benchmarks

Measured on **Apple M3 Max (128 GB Unified Memory)**:

| Stage / Metric | BF16 Baseline | 8-bit Quantized / Optimized | Improvement |
| :--- | :---: | :---: | :---: |
| **AR Model File Size** | 4.33 GB | **2.66 GB** | **-38.6%** memory |
| **AR Sampling Speed** | 59.4 tokens/s | **83.1 tokens/s** | **+39.9%** faster |
| **32-Step Full NAR Latents (170s)** | 363.09 s | **186.94 s** (Steel SDPA) | **1.94×** speedup |
| **Sampled Peak Memory Footprint** | ~11.0 GiB | **~10.2 GiB** | Stable (<16 GiB budget) |
| **Swap I/O** | 0 Bytes | 0 Bytes | Zero memory pressure |

*Both BF16 and 8-bit pipelines generate natural-ending songs with full 32 midpoint solver steps and bit-identical latent reconstruction.*

---

## Repository Structure

```
mlx-Yue/
├── src/lyra/                # Native MLX inference engine
│   ├── ar.py                # Autoregressive transformer with 8bit/BF16 KV-cache
│   ├── nar.py               # Non-autoregressive acoustic flow-matching solver
│   ├── vae.py               # Native FP32 Oobleck VAE decoder
│   ├── pipeline.py          # Unified YuE2Pipeline interface
│   ├── transcription/       # Native MLX SheetSage2/MERT2 transcription engine
│   └── commands.py          # CLI subcommands (generate, plan, cover, transcribe)
├── models/                  # Local model conversions and cache
├── tools/                   # Parity verification, calibration, and benchmarks
├── tests/                   # 100+ comprehensive automated unit & regression tests
└── reports/                 # Numerical parity, acceptance audits, and hardware logs
```

---

## Acknowledgments & Upstream

- [YuE / YuE2](https://github.com/multimodal-art-projection/YuE) by the Multimodal Art Projection (M-A-P) team.
- [MLX](https://github.com/ml-explore/mlx) by Apple Machine Learning Research.
- [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) by Stability AI (Oobleck VAE architecture).
- [SheetSage](https://github.com/chrisdonahue/sheetsage) & [MERT](https://github.com/m-a-p/MERT) for transcription foundation models.

## Reproduce the official full-song listening comparison

Use the installed project environment with the converted models and VAE paths in `models/paths.json`. The benchmark preserves official input text and ABC scores, evaluating both 32-step standard fidelity and 8-step real-time fast mode:

```bash
python tools/prepare_fullsong_benchmark.py
MLX_ENABLE_TF32=0 python tools/run_fullsong_benchmark.py      # 32-step full song generation
MLX_ENABLE_TF32=0 python tools/run_fast_8step_benchmark.py   # 8-step real-time fast mode synthesis
python tools/build_unified_showcase.py                       # Build interactive showcase
```

All 15 cases share one page, `docs/index.html`: six languages, all five generation modes, official cover/edit scores, and a score-recording-to-transcription-to-song workflow, complete with 3-channel side-by-side audio players and detailed timing/speedup analytics.
