"""Minimal command-line access to the complete offline Python generation API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from yue2.protocol import GenerationConfig
from yue2.storage import identity

from .measure import DEFAULT_MEMORY_BUDGET_GIB


def _pipeline(args, generation_config=None):
    from .pipeline import YuE2Pipeline
    from .conversion import MODEL_REPO, VAE_REPO

    return YuE2Pipeline.from_pretrained(
        args.model or MODEL_REPO, vae=args.vae or VAE_REPO,
        converted_dir=args.converted_dir, precision=args.precision,
        local_files_only=args.offline, memory_budget_gib=args.memory_budget_gib,
        vae_core_frames=args.vae_core_frames, progress=not args.quiet,
        generation_config=generation_config, require_ac=args.require_ac,
    )


def _request(args):
    if args.request and args.request_file:
        raise ValueError("Pass a positional request or --request, not both")
    filename = args.request or args.request_file
    request = json.loads(Path(filename).read_text()) if filename else {}
    for key in ("style", "lyrics", "seed", "cfg_scale", "id"):
        value = getattr(args, key, None)
        if value is not None:
            request[key] = value
    if args.lyrics_file:
        request["lyrics"] = Path(args.lyrics_file).read_bytes().decode("utf-8")
    if args.abc is not None:
        request["abc"] = Path(args.abc).read_bytes().decode("utf-8")
        request.pop("abc_path", None)
    if "abc_path" in request:
        if request.get("abc") is not None:
            raise ValueError("Pass abc or abc_path, not both")
        base = Path(filename).parent if filename else Path.cwd()
        request["abc"] = (base / request.pop("abc_path")).read_bytes().decode("utf-8")
    if args.mode is not None:
        request["cot"] = args.mode
    return request


def _save(result, output):
    result.save_artifacts(output)
    print(json.dumps({"output": str(output), "sample_rate": result.sample_rate,
                      "audio_seconds": len(result.audio) / result.sample_rate,
                      "truncated": result.truncated, "timing": result.timing}, indent=2))


def _resource_monitor(args):
    from .measure import ResourceMonitor

    output = Path(args.output)
    return ResourceMonitor(
        require_ac=args.require_ac,
        log_path=output.with_name(output.name + ".resources.jsonl"),
        report_path=output.with_name(output.name + ".resources.json"),
        metadata={
            "command": args.command,
            "memory_budget_gib": args.memory_budget_gib,
            "vae_core_frames": args.vae_core_frames,
            "precision": args.precision,
        },
    )


def _render_plan(pipe, plan, semantic_sampling=None):
    from .pipeline import SongResult, initial_noise

    start = time.perf_counter()
    semantic = pipe.generate_semantic(plan, sampling=semantic_sampling)
    noise = initial_noise(len(semantic.tokens), plan.request.seed)
    nar_start = time.perf_counter()
    latents = pipe.synthesize(semantic, noise=noise)
    nar_seconds = time.perf_counter() - nar_start
    vae_start = time.perf_counter()
    audio = pipe.decode(latents)
    config = pipe.effective_config(plan.request, semantic_sampling=semantic_sampling)
    stamp = identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights})
    timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
              "vae_seconds": time.perf_counter() - vae_start, "load": dict(pipe.load_timing),
              "e2e_seconds": time.perf_counter() - start}
    return SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, stamp, noise)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("music", "Agent generation/planning helper"),
                            ("abc", "Inspect, compare or strip chords from ABC"),
                            ("listen", "Build a local listening comparison page"),
                            ("transcribe", "Transcribe source audio with SheetSage2")):
        commands.add_parser(name, help=help_text, add_help=False).add_argument("arguments", nargs=argparse.REMAINDER)
    prep = commands.add_parser("prepare", help="Fetch pinned checkpoints and convert the generator")
    prep.add_argument("--source")
    prep.add_argument("--output", default="models/converted")
    prep.add_argument("--precision", choices=("bf16", "8bit", "4bit"), default="bf16")
    prep.add_argument("--offline", action="store_true")
    prep.add_argument("--cache-dir")
    doctor = commands.add_parser("doctor", help="Inspect runtime and local models without downloading")
    doctor.add_argument("--model", type=Path)
    doctor.add_argument("--vae", type=Path)
    doctor.add_argument("--verify-hashes", action="store_true")
    doctor.add_argument("--output", type=Path)
    batch = commands.add_parser("batch", help="Generate serial JSONL requests with verified resume")
    batch.add_argument("--input", required=True, type=Path)
    batch.add_argument("--concurrency", type=int, choices=[1], default=1)
    for name in ("generate", "plan", "render-plan", "replay", "cover"):
        command = commands.add_parser(name)
        command.add_argument("request", nargs="?", help="Request JSON, saved plan or song directory")
    cover = commands.choices["cover"]
    cover.add_argument("--audio", required=True, type=Path)
    cover.add_argument("--transcription-model", default="m-a-p/SheetSage2")
    cover.add_argument("--base-model", default="m-a-p/MERT-v2-FullSong")
    cover.add_argument("--cache-dir", default="models/hf-cache")
    cover.add_argument("--task", choices=("full", "melody-full", "melody-vocal"), default="melody-full")
    for name in ("generate", "plan", "render-plan", "replay", "batch", "cover"):
        command = commands.choices[name]
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--model", help="Converted directory or pinned source checkpoint")
        command.add_argument("--vae", help="Local default VAE checkpoint directory")
        command.add_argument("--converted-dir", default="models/converted")
        command.add_argument("--precision", choices=("bf16", "8bit", "4bit"), default="bf16")
        command.add_argument("--offline", action="store_true")
        command.add_argument("--quiet", action="store_true")
        command.add_argument("--memory-budget-gib", type=float, default=DEFAULT_MEMORY_BUDGET_GIB)
        command.add_argument("--vae-core-frames", type=int, default=256)
        command.add_argument("--require-ac", action="store_true")
        command.add_argument("--resume", action="store_true", help="Verify and reuse a completed song")
        if name in {"generate", "plan", "cover"}:
            command.add_argument("--request", dest="request_file")
            command.add_argument("--abc", "--abc-file", help="Supplied ABC file; bytes are preserved")
            command.add_argument("--mode", "--cot", choices=("full", "melody", "off"))
            command.add_argument("--style")
            lyrics = command.add_mutually_exclusive_group()
            lyrics.add_argument("--lyrics")
            lyrics.add_argument("--lyrics-file")
            command.add_argument("--seed", type=int)
            command.add_argument("--cfg-scale", type=float)
            command.add_argument("--id")
        if name == "replay":
            command.add_argument("--stage", choices=("synthesize", "decode"), default="decode")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    helpers = {"music": "run_yue2", "abc": "abc_tools", "listen": "listen", "transcribe": "transcribe"}
    if argv and argv[0] in helpers:
        import importlib
        module = importlib.import_module(".music_tools." + helpers[argv[0]], __package__)
        previous = sys.argv
        try:
            sys.argv = ["yue2-mlx " + argv[0], *argv[1:]]
            return module.main() or 0
        finally:
            sys.argv = previous
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "doctor":
        from .commands import doctor
        return doctor(args)
    if args.command == "batch":
        from .commands import batch
        return batch(args)
    if args.command == "cover":
        from .commands import cover
        return cover(args)
    if args.command == "prepare":
        from .conversion import fetch_models, prepare
        model, vae = fetch_models(cache_dir=args.cache_dir, local_files_only=args.offline)
        output = prepare(args.source or model, args.output, precision=args.precision)
        print(json.dumps({"model": str(output), "vae": str(vae)}))
        return 0
    if args.command in {"render-plan", "replay"} and not args.request:
        cli.error("A saved plan/song directory is required")
    if args.resume:
        if args.command != "generate":
            cli.error("--resume applies to completed generate or batch results")
        if (args.output / "result.json").is_file():
            from .commands import resume_result
            request = _request(args)
            generation = request.pop("generation_config", None)
            config = None if generation is None else GenerationConfig.from_dict(generation)
            with _pipeline(args, config) as pipe:
                print(json.dumps(resume_result(pipe, request, args.output), indent=2))
            return 0
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("Output directory must be empty; recordings are never silently overwritten")
    with _resource_monitor(args):
        if args.command in {"generate", "plan"}:
            request = _request(args)
            generation = request.pop("generation_config", None)
            config = None if generation is None else GenerationConfig.from_dict(generation)
            with _pipeline(args, config) as pipe:
                if args.command == "plan":
                    from .artifacts import save_plan_artifacts

                    abc_sampling = request.pop("abc_sampling", None)
                    semantic_sampling = request.pop("semantic_sampling", None)
                    plan = pipe.plan(**request, abc_sampling=abc_sampling)
                    effective = pipe.effective_config(
                        plan.request, abc_sampling, semantic_sampling,
                    )
                    save_plan_artifacts(
                        plan, args.output, GenerationConfig.from_dict(effective["generation"]),
                    )
                else:
                    result = pipe(**request)
            if args.command == "plan":
                print(json.dumps({"output": str(args.output), "truncated": plan.truncated,
                                  "timing": plan.timing}, indent=2))
            else:
                _save(result, args.output)
        elif args.command == "render-plan":
            from .artifacts import load_plan_artifacts

            plan, config = load_plan_artifacts(args.request)
            with _pipeline(args, config) as pipe:
                result = _render_plan(pipe, plan)
            _save(result, args.output)
        else:
            from .artifacts import load_artifacts
            from .pipeline import SongResult, initial_noise

            saved = load_artifacts(args.request)
            config = GenerationConfig.from_dict(saved.config["generation"])
            with _pipeline(args, config) as pipe:
                start = time.perf_counter()
                noise = saved.noise
                noise_source = "source_artifact" if noise is not None else "unavailable"
                latents = saved.latents
                nar_seconds = 0.0
                if args.stage == "synthesize":
                    if noise is None:
                        if saved.config.get("rng", {}).get("acoustic") != "numpy_pcg64_fp32_full_song_v1":
                            raise ValueError("Exact replay of legacy RNG requires retained noise.npy")
                        noise = initial_noise(
                            len(saved.semantic.tokens), saved.semantic.plan.request.seed,
                        )
                        noise_source = "regenerated_from_request_seed"
                    nar_start = time.perf_counter()
                    latents = pipe.synthesize(saved.semantic, noise=noise)
                    nar_seconds = time.perf_counter() - nar_start
                vae_start = time.perf_counter()
                audio = pipe.decode(latents)
                timing = {
                    "abc": saved.semantic.plan.timing,
                    "semantic": saved.semantic.timing,
                    "nar_seconds": nar_seconds,
                    "vae_seconds": time.perf_counter() - vae_start,
                    "load": dict(pipe.load_timing),
                    "e2e_seconds": time.perf_counter() - start,
                    "replay": {
                        "stage": args.stage,
                        "reused_latents": args.stage == "decode",
                    },
                }
                effective = pipe.effective_config(saved.semantic.plan.request)
                effective["replay"] = {
                    "stage": args.stage,
                    "source_identity": saved.result["identity"],
                    "source_config": saved.config,
                    "source_weights": saved.result["weights"],
                    "source_artifacts": saved.result["artifacts"],
                    "source_timing": saved.result["timing"],
                    "source_truncated": saved.result["truncated"],
                    "noise_source": noise_source,
                    "noise_used_for_synthesis": args.stage == "synthesize",
                }
                stamp = identity({
                    "request": saved.semantic.plan.request.to_dict(),
                    "config": effective,
                    "weights": pipe.weights,
                })
                result = SongResult(
                    audio, 48000, saved.semantic, latents, effective,
                    pipe.weights, timing, stamp, noise,
                )
            _save(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
