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
    import mlx.core as mx
    import psutil
    from .conversion import verify_conversion

    checks = {"macos": platform.system() == "Darwin", "metal": mx.metal.is_available()}
    report = {"platform": platform.platform(), "memory_gib": psutil.virtual_memory().total / 2**30,
              "backend": "mlx", "vae_backend": "mlx", "torch_required": False,
              "versions": {n: importlib.metadata.version(n) for n in ("lyra-yue2", "mlx", "mlx-lm")},
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
    if not rows or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("A batch requires nonempty JSONL with unique string request ids")
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
