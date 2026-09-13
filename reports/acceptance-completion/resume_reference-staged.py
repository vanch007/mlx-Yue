"""Finish the saved original reference stages in a fresh guarded MPS process."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from reference_song import CapturedPipeline
from lyra.measure import GPUExecution
from yue2.pipeline import SemanticResult, SongResult, SymbolicPlan
from yue2.protocol import CODEC_SIZE, SongRequest
from yue2.storage import identity, sha256_file, write_json
from staged_reference import synthesize

source = ROOT / "outputs/acceptance-completion/reference-full-song-bounded"
stage = source / "semantic-stage"
output = ROOT / "outputs/acceptance-completion/reference-full-song-staged"
if output.exists():
    raise FileExistsError(output)
output.mkdir(parents=True)
request_path = ROOT / "outputs/precision-comparison/full-song-bf16/request.json"
request = SongRequest(**json.loads(request_path.read_text()))
source_report = json.loads((source / "reference.json").read_text())
assert source_report["request_sha256"] == sha256_file(request_path)
plan = SymbolicPlan.load(stage)
assert plan.request.to_dict() == request.to_dict()
metadata = json.loads((stage / "semantic.json").read_text())
assert metadata["tokens_sha256"] == sha256_file(stage / "semantic.npy")
tokens = np.load(stage / "semantic.npy", allow_pickle=False)
assert tokens.ndim == 1 and tokens.dtype.kind in "iu" and len(tokens) > 0
assert tokens.min() >= 0 and tokens.max() < CODEC_SIZE
assert not metadata["truncated"] and not plan.truncated
semantic = SemanticResult(plan, tokens.tolist(), metadata["timing"], metadata["truncated"])
model = ROOT / "models/hf-cache/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19"
vae = json.loads((ROOT / "models/paths.json").read_text())["vae"]
report = {"status": "running", "backend": "original_torch_mps", "listening_status": "pending",
    "request_sha256": sha256_file(request_path), "script_sha256": sha256_file(__file__),
    "reference_helper_sha256": sha256_file(ROOT / "tools/reference_song.py"),
    "resumed_original_stage": str(stage), "semantic_sha256": sha256_file(stage / "semantic.npy"),
    "stage_receipt_sha256": sha256_file(stage / "semantic.json"),
    "source_reference_sha256": sha256_file(source / "reference.json"),
    "staging_helper_sha256": sha256_file(Path(__file__).with_name("staged_reference.py")),
    "comparison": "Original AR output resumed with original BF16 modules; phase-specific weight residency; same pinned model, original chunks/noise, 32 midpoint steps; not a speed benchmark"}
write_json(output / "reference.json", report)
start = time.perf_counter()
try:
    with GPUExecution(backend="mps", memory_budget_gib=16, require_ac=True,
                      resource_path=output / "resources.jsonl") as guard:
        def cancelled():
            torch.mps.synchronize()
            torch.mps.empty_cache()
            guard.check()
            return False

        with CapturedPipeline(model, vae, device="mps", backend="torch-eager",
                              memory_budget_gib=16, vae_core_frames=256) as pipe:
            config = pipe.effective_config(request)
            report["weights"] = pipe.weights
            write_json(output / "reference.json", report)
            acoustic_start = time.perf_counter()
            latents = synthesize(model, semantic, pipe.generation_config, guard)
            np.save(output / "latent.npy", latents)
            nar_seconds = time.perf_counter() - acoustic_start
            cancelled()
            decode_start = time.perf_counter()
            audio = pipe.decode(latents)
            timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                "vae_seconds": time.perf_counter() - decode_start, "load": dict(pipe.load_timing),
                "e2e_seconds": time.perf_counter() - start, "scope": "resumed acoustic/decode process only"}
            song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing,
                              identity({"request": request.to_dict(), "config": config, "weights": pipe.weights}))
            guard.check()
            song.save_artifacts(output / "song")
            report.update(audio_seconds=len(audio) / 48000, truncated=song.truncated, timing=timing)
    report["status"] = "pass_execution_listening_pending"
except BaseException as error:
    report.update(status="fail", error=f"{type(error).__name__}: {error}")
    raise
finally:
    report["seconds"] = time.perf_counter() - start
    write_json(output / "reference.json", report)
