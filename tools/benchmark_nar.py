"""Compare complete acoustic synthesis using integrity-checked saved inputs.

Run baseline and auto separately, with the same memory budget and no competing
GPU workload. Baseline restores the original FP32 operand promotion. Neither
variant changes tokens, noise, 32 midpoint steps, or query visibility.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from lyra.ar import load_ar
from lyra.artifacts import load_artifacts
from lyra.measure import GPUExecution
from lyra.nar import CachedNAR, _chunks_with_supplied_noise, load_nar
from yue2.storage import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/converted"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", choices=("baseline", "auto"), default="auto")
    parser.add_argument("--memory-budget-gib", type=float, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = load_artifacts(args.source)
    chunks = _chunks_with_supplied_noise(saved.semantic.plan.prefix, saved.semantic.tokens,
                                        saved.semantic.plan.request.seed, 24576, saved.noise)
    context = patch("lyra.ar._full_attention_requires_promotion", return_value=True) if args.attention == "baseline" else nullcontext()
    cache_hash = hashlib.sha256()
    prefill_seconds = solve_seconds = 0.0
    latents = []
    with context, GPUExecution(memory_budget_gib=args.memory_budget_gib, require_ac=True,
                               resource_path=args.output / "resources.jsonl") as guard:
        ar = load_ar(args.model, precision="bf16")
        model = load_nar(args.model, ar, verify=False)
        for chunk_index, chunk in enumerate(chunks):
            start = time.perf_counter()
            engine = CachedNAR(model, chunk, query_chunk_size=256, cancelled=lambda: guard.check())
            prefill_seconds += time.perf_counter() - start
            for pair in engine.cache:
                for item in pair:
                    cache_hash.update(np.array(item.astype(mx.float32)).tobytes())
            def progress(completed, total):
                print(json.dumps({"chunk": chunk_index, "step": completed, "total": total}), flush=True)
            start = time.perf_counter()
            latents.append(engine.solve(32, cancelled=lambda: guard.check(), on_progress=progress))
            solve_seconds += time.perf_counter() - start
            engine.close()
        result = np.concatenate(latents)
        np.save(args.output / "latent.npy", result)
        error = result - saved.latents
        report = {"attention": args.attention, "memory_budget_gib": args.memory_budget_gib,
                  "device": mx.device_info(), "source": str(args.source.resolve()),
                  "source_result_sha256": sha256_file(args.source / "result.json"),
                  "frames": len(result), "steps": 32, "query_chunk_size": 256,
                  "prefill_seconds": prefill_seconds, "solve_seconds": solve_seconds,
                  "acoustic_seconds": prefill_seconds + solve_seconds,
                  "conditioning_cache_sha256": cache_hash.hexdigest(),
                  "max_abs_vs_saved": float(np.max(np.abs(error))),
                  "equal_fraction_vs_saved": float(np.mean(result == saved.latents)),
                  "relative_rms_vs_saved": float(np.sqrt(np.mean(error ** 2) / np.mean(saved.latents ** 2))),
                  "mlx_peak_gib": mx.get_peak_memory() / 2**30,
                  "code_sha256": {name: sha256_file(Path("src/lyra") / name) for name in ("ar.py", "nar.py")}}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
