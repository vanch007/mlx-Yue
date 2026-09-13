"""Measure full-song velocity with identical checkpoint, conditioning and state.

Experiments do not change production defaults. Native BF16 SDPA is eligible only
after source inspection and numerical checks on the actual GPU architecture.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from lyra.ar import load_ar
from lyra.measure import GPUExecution
from lyra.nar import CachedNAR, _attention, _chunks_with_supplied_noise, load_nar
from lyra.artifacts import load_artifacts


def native_attention(q, k, v, *, causal=False, query_chunk_size=None):
    outputs = []
    block = q.shape[2] if query_chunk_size is None else query_chunk_size
    for start in range(0, q.shape[2], block):
        end = min(start + block, q.shape[2])
        outputs.append(mx.fast.scaled_dot_product_attention(
            q[..., start:end, :], k[..., :end, :] if causal else k,
            v[..., :end, :] if causal else v, scale=q.shape[-1] ** -.5,
            mask="causal" if causal else None,
        ))
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def promoted_attention(*args, **kwargs):
    with patch("lyra.ar._full_attention_requires_promotion", return_value=True):
        return _attention(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/converted"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=("attention", "cache", "alternating"), default="attention")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = load_artifacts(args.source)
    chunk = _chunks_with_supplied_noise(saved.semantic.plan.prefix, saved.semantic.tokens,
                                       saved.semantic.plan.request.seed, 24576, saved.noise)[0]
    records = []
    with GPUExecution(memory_budget_gib=32, require_ac=True,
                      resource_path=args.output / "resources.jsonl") as guard:
        ar = load_ar(args.model, precision="bf16")
        model = load_nar(args.model, ar, verify=False)
        start = time.perf_counter()
        engine = CachedNAR(model, chunk, query_chunk_size=256, cancelled=lambda: guard.check())
        print(json.dumps({"prefill_seconds": time.perf_counter() - start,
                          "frames": len(chunk.noise), "device": mx.device_info()}), flush=True)
        state = mx.array(chunk.noise, dtype=mx.bfloat16)
        reference = None
        variants = (("fp32-tile256", 256, promoted_attention, 128),
                    ("fp32-tile1024", 1024, promoted_attention, 128),
                    ("fp32-full", None, promoted_attention, 128),
                    ("bf16-tile256", 256, native_attention, 128),
                    ("bf16-full", None, native_attention, 128))
        if args.study == "cache":
            variants = [(f"cache-{mib}MiB", 256, _attention, mib) for mib in (128, 512, 1024, 2048)]
        elif args.study == "alternating":
            variants = [(f"{name}-{i}", 256, implementation, 128) for i in (1, 2)
                        for name, implementation in (("fp32", promoted_attention), ("bf16", native_attention))]
        for name, tile, implementation, cache_mib in variants:
            mx.clear_cache()
            mx.set_cache_limit(cache_mib * 2**20)
            engine._attend = lambda q, k, v, causal=False: implementation(q, k, v, causal=causal, query_chunk_size=tile)
            if args.study == "alternating":
                mx.eval(engine.velocity(state, 0.0))
            samples = []
            for _ in range(3):
                guard.check()
                start = time.perf_counter()
                value = engine.velocity(state, 0.0)
                mx.eval(value)
                samples.append(time.perf_counter() - start)
            actual = np.array(value.astype(mx.float32))
            if reference is None:
                reference = actual.copy()
                np.save(args.output / "reference-velocity.npy", reference)
            error = actual - reference
            record = {"variant": name, "seconds": samples,
                      "max_abs": float(np.max(np.abs(error))),
                      "relative_rms": float(np.sqrt(np.mean(error ** 2) / np.mean(reference ** 2))),
                      "equal_fraction": float(np.mean(actual == reference)),
                      "mlx_peak_gib": mx.get_peak_memory() / 2**30}
            records.append(record)
            (args.output / "profile.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        engine.close()


if __name__ == "__main__":
    main()
