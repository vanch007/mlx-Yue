#!/usr/bin/env python3
"""Compare ODE steps on verified identical semantic IDs and retained solver noise."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

from generate_music import model_paths, native_modules, project_root, summarize, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", nargs="+", type=int, default=[8, 32])
    p.add_argument("--project")
    p.add_argument("--model")
    p.add_argument("--vae")
    p.add_argument("--allow-battery", action="store_true")
    p.add_argument("--memory-budget-gib", type=float, default=16)
    args = p.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("Output must be a new absolute directory")
    if len(set(args.steps)) != len(args.steps) or any(n < 1 for n in args.steps):
        raise ValueError("Steps must be distinct positive integers")
    root = project_root(args.project)
    native_modules(root)
    from lyra import YuE2Pipeline
    from lyra.artifacts import load_artifacts
    from lyra.measure import ResourceMonitor
    from lyra.pipeline import SongResult
    from lyra.music_tools.listen import build
    from yue2.protocol import GenerationConfig
    from yue2.storage import identity

    saved = load_artifacts(args.source)
    if saved.noise is None:
        raise ValueError("Controlled step comparison requires retained noise.npy")
    model, vae = model_paths(root, args.model, args.vae)
    config = GenerationConfig.from_dict(saved.config["generation"])
    args.output.mkdir(parents=True)
    rows, sources = [], []
    load_start = time.perf_counter()
    with YuE2Pipeline.from_pretrained(
        model, vae=vae, precision=saved.config.get("ar_precision", "bf16"),
        local_files_only=True, require_ac=not args.allow_battery,
        memory_budget_gib=args.memory_budget_gib, generation_config=config,
    ) as pipe:
        load_seconds = time.perf_counter() - load_start
        if pipe.weights != saved.result["weights"]:
            raise ValueError("Loaded model/VAE identity differs from source; not a controlled step comparison")
        for steps in args.steps:
            pipe.generation_config = replace(config, ode_steps=steps)
            dest = args.output / f"steps-{steps}"
            resources = args.output / f"steps-{steps}.resources.json"
            with ResourceMonitor(require_ac=not args.allow_battery, report_path=resources):
                start = time.perf_counter()
                latent = pipe.synthesize(saved.semantic, noise=saved.noise)
                nar = time.perf_counter() - start
                vae_start = time.perf_counter()
                audio = pipe.decode(latent)
                vae_seconds = time.perf_counter() - vae_start
                compute = time.perf_counter() - start
            effective = pipe.effective_config(saved.semantic.plan.request)
            effective["step_comparison"] = {
                "source_identity": saved.result["identity"], "source_config": saved.config,
                "source_artifacts": saved.result["artifacts"],
                "reused": ["ABC", "semantic_ids", "noise"], "timing_scope": "NAR + VAE only",
            }
            stamp = identity({"request": saved.semantic.plan.request.to_dict(), "config": effective, "weights": pipe.weights})
            # SongResult requires the original stage timings for provenance.
            # Current compute/RTF excludes those reused stages explicitly.
            timing = {"abc": saved.semantic.plan.timing, "semantic": saved.semantic.timing,
                      "nar_seconds": nar, "vae_seconds": vae_seconds, "e2e_seconds": compute,
                      "scope": "resynthesis_only", "reused_stages": ["abc", "semantic"],
                      "shared_load_seconds": load_seconds}
            result = SongResult(audio, 48000, saved.semantic, latent, effective, pipe.weights, timing, stamp, saved.noise)
            result.save_artifacts(dest)
            row = summarize(dest, compute, resources, scope="NAR + VAE; excludes reused AR, shared model load and saving")
            row.update(ode_steps=steps, shared_load_seconds=load_seconds)
            rows.append(row)
            sources.append(dest)
            write(args.output / f"steps-{steps}.measurement.json", row)
    write(args.output / "comparison.json", {"scope": "controlled resynthesis; not full generation RTF", "results": rows})
    build(sources, args.output / "listen")
    print(args.output / "listen/index.html")
    return int(any(not r["checks"][k] for r in rows for k in ("artifacts_verified", "finite_audio", "48khz_stereo", "non_silent")))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
