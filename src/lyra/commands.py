"""Local diagnostics and serial request orchestration for the MLX backend."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

from yue2.protocol import GenerationConfig, SongRequest
from yue2.storage import identity, model_identity, verify_result, write_json


def resume_result(pipe, request, output):
    fields = {k: v for k, v in request.items() if k not in {"abc_sampling", "semantic_sampling"}}
    native = pipe._request(**fields)
    config = pipe.effective_config(native, request.get("abc_sampling"), request.get("semantic_sampling"))
    expected = identity({"request": native.to_dict(), "config": config, "weights": pipe.weights})
    result = verify_result(output, expected)
    return {"resumed": True, "output": str(output), "identity": result["identity"],
            "truncated": result["truncated"]}


def doctor(args):
    import psutil
    from .conversion import verify_conversion
    from .runtime import runtime_status

    runtime = runtime_status()
    checks = {"macos": runtime["system"] == "Darwin", "metal": runtime["metal"],
              "supported_runtime": runtime["supported"]}
    report = {"platform": platform.platform(), "memory_gib": psutil.virtual_memory().total / 2**30,
              "runtime": runtime,
              "backend": "mlx", "vae_backend": "mlx", "torch_required": False,
              "versions": {n: importlib.metadata.version(n) for n in ("mlx-yue", "mlx", "mlx-lm")},
              "checks": checks}
    for kind in ("model", "vae"):
        path = getattr(args, kind)
        if path is None:
            continue
        try:
            if kind == "model":
                data = verify_conversion(path) if args.verify_hashes else json.loads((path / "conversion.json").read_text())
            else:
                data = model_identity(path) if args.verify_hashes else json.loads((path / "config.json").read_text())
            checks[kind] = True
            report[kind] = {"path": str(path.resolve()), "identity_checked": args.verify_hashes,
                            "metadata": data}
        except (OSError, ValueError) as error:
            checks[kind] = False
            report[kind] = {"error": str(error)}
    report["status"] = "pass" if all(checks.values()) else "fail"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return int(report["status"] == "fail")


def batch(args):
    from .cli import _pipeline

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    ids = [row.get("id") for row in rows]
    if (not rows or any(not isinstance(i, str) for i in ids)
            or len({i.casefold() for i in ids}) != len(ids)):
        raise ValueError("A batch requires nonempty JSONL with case-insensitively unique string request ids")
    if any(i.casefold() == "batch.json" for i in ids):
        raise ValueError("batch.json is reserved for the batch manifest")
    prepared = []
    for row in rows:
        data = dict(row)
        config = GenerationConfig.from_dict(data.pop("generation_config", {}))
        if "abc_path" in data:
            if data.get("abc") is not None:
                raise ValueError("Pass abc or abc_path, not both")
            data["abc"] = (args.input.parent / data.pop("abc_path")).read_bytes().decode("utf-8")
        fields = {k: v for k, v in data.items() if k not in {"abc_sampling", "semantic_sampling"}}
        if "tags" in fields:
            if "style" in fields and fields["style"] != fields["tags"]:
                raise ValueError("style and tags disagree")
            fields["style"] = fields.pop("tags")
        SongRequest(**fields)  # Reject unsafe ids and invalid requests before writing or loading weights.
        prepared.append((data, config))
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use an empty batch directory or --resume")
    receipts = []
    with _pipeline(args) as pipe:
        for data, config in prepared:
            pipe.generation_config = config
            destination = args.output / data["id"]
            try:
                if args.resume and (destination / "result.json").exists():
                    receipt = resume_result(pipe, data, destination)
                else:
                    if destination.exists() and any(destination.iterdir()):
                        raise FileExistsError(f"Incomplete/nonempty output requires a fresh destination: {destination}")
                    song = pipe(**data)
                    receipt = song.save_artifacts(destination)
                receipts.append({"id": data["id"], "status": "complete", "identity": receipt["identity"],
                                 "truncated": receipt["truncated"], "resumed": receipt.get("resumed", False)})
            except Exception as error:
                receipts.append({"id": data["id"], "status": "failed", "error": str(error),
                                 "error_type": type(error).__name__})
            write_json(args.output / "batch.json", {"expected": len(rows), "results": receipts})
            if not args.quiet:
                print(json.dumps(receipts[-1]), file=sys.stderr)
    failed = sum(r["status"] == "failed" for r in receipts)
    print(json.dumps({"output": str(args.output), "requests": len(rows), "failed": failed}))
    return int(failed > 0)


def cover(args):
    """Transcribe the complete recording, then render the retained score in MLX."""
    import gc
    import mlx.core as mx
    from .cli import _pipeline, _request, _resource_monitor, _save
    from .measure import GPUExecution
    from .transcription.pipeline import transcribe

    request = _request(args)
    if request.get("abc") is not None or args.resume:
        raise ValueError("Cover takes source audio and a fresh output; use generate for an existing ABC")
    mode = "full" if args.task == "full" else "melody"
    if request.get("cot", mode) != mode:
        raise ValueError("Cover mode must match the transcription task")
    request["cot"] = mode
    generation = GenerationConfig.from_dict(request.pop("generation_config", {}))
    # Validate text, seed and mode before spending time on transcription.
    SongRequest(**{k: v for k, v in request.items() if k not in {"abc_sampling", "semantic_sampling"}})
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Cover requires a fresh output directory")
    with _resource_monitor(args):
        with GPUExecution(memory_budget_gib=args.memory_budget_gib) as guard:
            def cancelled():
                guard.check()
                return False

            transcription = transcribe(
                args.audio, args.output / "transcription", model_path=args.transcription_model,
                base_model=args.base_model, offline=args.offline, cache_dir=args.cache_dir,
                task=args.task, cancelled=cancelled,
            )
            guard.check()
        gc.collect()
        mx.clear_cache()
        if transcription["status"] != "complete" or transcription["truncated"]:
            raise ValueError("Transcription is incomplete; inspect its artifacts before using the score")
        request["abc"] = (args.output / "transcription" / "score.abc").read_bytes().decode("utf-8")
        write_json(args.output / "request.json", request)
        with _pipeline(args, generation) as pipe:
            result = pipe(**request)
        _save(result, args.output / "song")
    write_json(args.output / "cover.json", {
        "source_audio_sha256": transcription["source_audio_sha256"],
        "transcription": "transcription/result.json", "song": "song/result.json",
        "backend": "mlx", "truncated": result.truncated,
    })
    return int(any(result.truncated.values()))
