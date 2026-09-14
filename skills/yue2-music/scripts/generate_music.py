#!/usr/bin/env python3
"""Validate/profile native YuE2 requests; optionally execute the project CLI offline."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROFILES = {"fast": (8, "8bit"), "standard": (32, "8bit"), "reference": (32, "bf16")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def project_root(value=None):
    root = Path(value or os.environ.get("YUE2_PROJECT", Path(__file__).resolve().parents[3])).expanduser().resolve()
    if not (root / "src/lyra/cli.py").is_file():
        raise ValueError(f"Not an mlx-Yue checkout: {root}; supply --project")
    return root


def native_modules(root):
    # Only local protocol/helpers, no model download or GPU allocation in preflight.
    sys.path[:0] = [str(root / "src"), str(root / "vendor/yue/src")]


def configure(data, request_path, profile="standard", preview_seconds=None, action="generate", task="melody-full"):
    from yue2.protocol import GenerationConfig, SongRequest, resolve_sampling

    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    data = copy.deepcopy(data)
    if "tags" in data:
        tags = data.pop("tags")
        if "style" in data and data["style"] != tags:
            raise ValueError("style and tags disagree")
        data["style"] = tags
    if "abc_path" in data:
        if data.get("abc") is not None:
            raise ValueError("Supply abc or abc_path, not both")
        data["abc"] = (Path(request_path).resolve().parent / data.pop("abc_path")).read_bytes().decode("utf-8")
    mode = "full" if action != "cover" or task == "full" else "melody"
    data.setdefault("cot", mode)
    if action == "cover" and (data.get("abc") is not None or data["cot"] != mode):
        raise ValueError("Cover requires audio, no supplied ABC, and cot matching --task")
    if action == "plan" and data["cot"] == "off":
        raise ValueError("off has no symbolic score to plan; use generate")
    config = data.setdefault("generation_config", {})
    config.setdefault("ode_steps", PROFILES[profile][0])  # Explicit user settings win.
    if preview_seconds is not None:
        if not math.isfinite(preview_seconds) or preview_seconds <= 0:
            raise ValueError("preview_seconds must be finite and positive")
        cap = max(1, math.ceil(preview_seconds * 25))
        sampling = data.setdefault("semantic_sampling", {})
        sampling["max_tokens"] = cap
        sampling["min_tokens"] = min(sampling.get("min_tokens", config.get("semantic", {}).get("min_tokens", 200)), cap)
    gc = GenerationConfig.from_dict(config)
    resolve_sampling(data.get("abc_sampling"), gc.abc)
    resolve_sampling(data.get("semantic_sampling"), gc.semantic)
    fields = {k: v for k, v in data.items() if k not in {"generation_config", "abc_sampling", "semantic_sampling"}}
    req = SongRequest(**fields)  # Reject unsupported ACE-Step fields and invalid counts.
    if req.abc:
        from lyra.music_tools.abc_tools import parse_abc
        score = parse_abc(req.abc)
        if req.cot == "melody" and any(v.chords for v in score.voices.values()):
            raise ValueError("melody ABC contains chords; run mlx-yue abc strip-chords first")
    return {**req.to_dict(), **{k: data[k] for k in ("generation_config", "abc_sampling", "semantic_sampling") if k in data}}


def model_paths(root, model=None, vae=None):
    paths = read(root / "models/paths.json") if (root / "models/paths.json").is_file() else {}
    def resolve(value):
        path = Path(value).expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    model = resolve(model or paths.get("model", "models/converted"))
    vae = resolve(vae or paths.get("vae", "models/vae"))
    if not (model / "conversion.json").is_file() or not (vae / "config.json").is_file():
        raise ValueError("Local converted generator/VAE missing; see references/models-and-setup.md")
    return model, vae


def summarize(song_dir, wall_seconds, resources_path=None, scope="full_cli_including_load_and_save"):
    import numpy as np
    import soundfile as sf
    from yue2.storage import verify_result

    result = verify_result(song_dir)
    audio, rate = sf.read(str(Path(song_dir) / "audio.flac"), dtype="float32", always_2d=True)
    seconds = len(audio) / rate
    finite = bool(np.isfinite(audio).all())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if finite and len(audio) else None
    checks = {"artifacts_verified": True, "finite_audio": finite,
              "48khz_stereo": rate == 48000 and audio.shape[1] == 2,
              "duration_matches_receipt": abs(seconds - result["audio_seconds"]) <= 1 / rate,
              "non_silent": rms is not None and rms > 1e-6,
              "natural_end": not any(result["truncated"].values())}
    resources = read(resources_path) if resources_path and Path(resources_path).is_file() else {}
    peaks = resources.get("sampled_maxima", {})
    footprint = peaks.get("physical_footprint_bytes")
    return {"song": str(Path(song_dir).resolve()), "checks": checks,
            "technical_status": "pass" if all(checks.values()) else "needs_review",
            "audio_seconds": seconds, "wall_seconds": wall_seconds, "timing_scope": scope,
            "rtf": wall_seconds / seconds if seconds else None,
            "sampled_peak_footprint_gib": footprint / 2**30 if footprint is not None else None,
            "resource_report": str(resources_path) if resources else None,
            "rms": rms, "peak": float(np.abs(audio).max()) if finite and len(audio) else None,
            "clipping_fraction": float(np.mean(np.abs(audio) >= 1)) if finite and len(audio) else None,
            "timing": result["timing"], "truncated": result["truncated"], "identity": result["identity"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="New absolute run directory")
    p.add_argument("--project")
    p.add_argument("--profile", choices=PROFILES, default="standard")
    p.add_argument("--action", choices=("generate", "plan", "cover"), default="generate")
    p.add_argument("--audio", type=Path)
    p.add_argument("--task", choices=("full", "melody-full", "melody-vocal"), default="melody-full")
    p.add_argument("--model")
    p.add_argument("--vae")
    p.add_argument("--precision", choices=("bf16", "8bit", "4bit"))
    p.add_argument("--memory-budget-gib", type=float, default=16)
    p.add_argument("--vae-core-frames", type=int, default=256)
    p.add_argument("--preview-seconds", type=float, help="Explicit capped diagnostic only; not a full song duration")
    p.add_argument("--allow-battery", action="store_true")
    p.add_argument("--execute", action="store_true", help="Without this flag validate and print the request only")
    args = p.parse_args(argv)
    root = project_root(args.project)
    native_modules(root)
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError("--output must be an unused absolute path")
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0 or args.vae_core_frames < 1:
        raise ValueError("Invalid memory budget or VAE core frames")
    if args.action == "cover" and (args.audio is None or not args.audio.is_file()):
        raise ValueError("Cover requires an existing --audio file")
    if args.action != "cover" and args.audio is not None:
        raise ValueError("--audio is only supported with --action cover")
    data = configure(read(args.request), args.request, args.profile, args.preview_seconds, args.action, args.task)
    precision = args.precision or PROFILES[args.profile][1]
    decision = {"profile": args.profile, "precision": precision, "effective_request": data,
                "preview_cap_seconds": args.preview_seconds, "require_ac": not args.allow_battery}
    if not args.execute:
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    model, vae = model_paths(root, args.model, args.vae)
    if not (model / f"ar-{precision}.safetensors").is_file():
        raise ValueError(f"AR {precision} weights missing; no automatic precision fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "effective-request.json", data)
    write(args.output / "decision.json", decision)
    destination = args.output / args.action
    cmd = [sys.executable, "-m", "lyra.cli", args.action, str(args.output / "effective-request.json"),
           "--output", str(destination), "--model", str(model), "--vae", str(vae), "--precision", precision,
           "--offline", "--memory-budget-gib", str(args.memory_budget_gib), "--vae-core-frames", str(args.vae_core_frames)]
    if not args.allow_battery:
        cmd.append("--require-ac")
    if args.action == "cover":
        cmd.extend(["--audio", str(args.audio.resolve()), "--task", args.task,
                    "--cache-dir", str(root / "models/hf-cache")])
    env = os.environ.copy()
    env.setdefault("MLX_ENABLE_TF32", "0")
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "vendor/yue/src"), env.get("PYTHONPATH", "")])
    write(args.output / "invocation.json", {"argv": cmd, "cwd": str(root), "timing_scope": "CLI subprocess including model load and artifact save"})
    start = time.perf_counter()
    run = subprocess.run(cmd, cwd=root, env=env, check=False)
    wall = time.perf_counter() - start
    if run.returncode:
        write(args.output / "failure.json", {"returncode": run.returncode, "wall_seconds": wall})
        return run.returncode
    if args.action == "plan":
        from lyra.artifacts import load_plan_artifacts
        plan, _ = load_plan_artifacts(destination)
        summary = {"output": str(destination), "truncated": plan.truncated, "wall_seconds": wall}
        status = int(plan.truncated)
    else:
        song = destination / "song" if args.action == "cover" else destination
        summary = summarize(song, wall, destination.with_name(destination.name + ".resources.json"))
        status = int(summary["technical_status"] != "pass")
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
