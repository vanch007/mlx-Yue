"""Fixed-corpus real-checkpoint workloads; success never implies listening approval."""
from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from lyra import YuE2Pipeline
from lyra.measure import GPUExecution, memory_snapshot
from yue2.protocol import SongRequest
from yue2.storage import identity, sha256_file, write_json


def request(case):
    return {key: case[key] for key in SongRequest.__dataclass_fields__ if key in case}


def _resource_evidence(output, stem="resources"):
    result = {}
    for suffix in ("jsonl", "json"):
        path = Path(output) / f"{stem}.{suffix}"
        if path.is_file():
            result[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _input_hashes(args):
    result = {"corpus": sha256_file(args.corpus), "script": sha256_file(__file__)}
    candidates = {
        "conversion": Path(args.model) / "conversion.json",
        "model_config": Path(args.model) / "config.json",
        "pipeline": Path(args.model) / "pipeline.json",
        "vae_config": Path(args.vae) / "config.json",
    }
    for name, path in candidates.items():
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("modes", "full", "sustain"))
    parser.add_argument("--model", default="models/converted")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--precision", choices=("bf16", "8bit", "4bit"), default="bf16")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus", default="validation/corpus.json", type=Path)
    parser.add_argument("--require-ac", action="store_true")
    parser.add_argument("--memory-budget-gib", type=float, default=16)
    parser.add_argument("--vae-core-frames", type=int, default=256)
    args = parser.parse_args()
    if args.vae_core_frames < 1:
        parser.error("--vae-core-frames must be a positive integer")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Use a new output directory; acceptance evidence is never overwritten"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(args.corpus.read_text())
    cases = {case["id"]: case for case in corpus["cases"]}
    workloads = []
    if args.workload == "modes":
        for mode, source in (
            ("full", "generated"),
            ("full", "supplied"),
            ("melody", "generated"),
            ("melody", "supplied"),
            ("off", "none"),
        ):
            value = request(cases["city_lights"])
            value.update(cot=mode, id=f"{mode}_{source}")
            if source == "supplied":
                value["abc"] = Path(
                    "validation/score.abc" if mode == "full" else "validation/melody.abc"
                ).read_bytes().decode("utf-8")
            if mode == "melody" and source == "supplied":
                value["cfg_scale"] = 1.2
            value["semantic_sampling"] = {
                "max_tokens": 400,
                "min_tokens": 200,
                "top_k": 80,
            }
            workloads.append(value)
    else:
        for index in range(3 if args.workload == "sustain" else 1):
            value = request(cases["tonight_awake"])
            value["id"] = f"tonight_awake_{index + 1}"
            workloads.append(value)

    input_hashes = _input_hashes(args)
    invocation = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "workload": args.workload,
        "precision": args.precision,
        "requests": workloads,
        "request_set_sha256": identity(workloads),
        "input_sha256": input_hashes,
        "runtime": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            **{
                name: version(name)
                for name in (
                    "lyra-yue2",
                    "mlx",
                    "mlx-lm",
                    "transformers",
                    "numpy",
                )
            },
        },
        "execution_config": {
            "backend": "mlx",
            "memory_budget_gib": args.memory_budget_gib,
            "vae_core_frames": args.vae_core_frames,
            "require_ac": args.require_ac,
            "resident_pipeline": True,
            "serialized_runs": True,
            "ar_cache_dtype": "bfloat16",
            "nar_dtype": "bfloat16",
            "vae_dtype": "float32",
        },
    }
    invocation["invocation_sha256"] = identity(invocation)
    write_json(args.output / "invocation.json", invocation)
    plan = {
        "workload": args.workload,
        "precision": args.precision,
        "corpus_sha256": input_hashes["corpus"],
        "requests": workloads,
        "request_set_sha256": invocation["request_set_sha256"],
        "invocation_sha256": invocation["invocation_sha256"],
        "listening_status": "not_evaluated",
        "require_ac": args.require_ac,
        "memory_budget_gib": args.memory_budget_gib,
        "effective_vae_core_frames": args.vae_core_frames,
        "full_song_gate_applicable": args.workload != "modes",
    }
    write_json(args.output / "workload.json", plan)

    reports, releases = [], []
    report = {
        **plan,
        "status": "running",
        "construction_seconds": None,
        "effective_pipeline_config": None,
        "runs": reports,
        "after_result_release": releases,
        "overall_resources": None,
        "resource_evidence": {},
        "full_duration_gate": None,
    }
    write_json(args.output / "report.json", report)
    construction_seconds = None
    active_row = None
    overall = None
    try:
        # One resident pipeline and one process cover the complete serial workload.
        with GPUExecution(
            backend="mlx",
            memory_budget_gib=args.memory_budget_gib,
            resource_path=args.output / "resources.jsonl",
            require_ac=args.require_ac,
        ) as overall:
            overall.check()
            start = time.perf_counter()
            with YuE2Pipeline.from_pretrained(
                args.model,
                vae=args.vae,
                precision=args.precision,
                local_files_only=True,
                memory_budget_gib=args.memory_budget_gib,
                vae_core_frames=args.vae_core_frames,
            ) as pipe:
                construction_seconds = time.perf_counter() - start
                report["construction_seconds"] = construction_seconds
                report["effective_pipeline_config"] = {
                    "precision": pipe.precision,
                    "memory_budget_gib": pipe.memory_budget_gib,
                    "vae_core_frames": pipe.vae_core_frames,
                    "query_chunk_size": pipe.query_chunk_size,
                    "execution_guard": "GPUExecution",
                }
                write_json(args.output / "report.json", report)
                for value in workloads:
                    overall.check()
                    run_id = value["id"]
                    active_row = {
                        "id": run_id,
                        "request_sha256": identity(value),
                        "status": "running",
                        "progress": {
                            "abc_output_tokens": 0,
                            "semantic_output_tokens": 0,
                        },
                        "resource_files": {
                            "samples": f"{run_id}.resources.jsonl",
                            "final": f"{run_id}.resources.json",
                        },
                    }
                    reports.append(active_row)
                    write_json(args.output / f"{run_id}.measurement.json", active_row)
                    write_json(args.output / "report.json", report)
                    with GPUExecution(
                        backend="mlx",
                        memory_budget_gib=args.memory_budget_gib,
                        resource_path=args.output / f"{run_id}.resources.jsonl",
                        require_ac=args.require_ac,
                    ) as run_guard:
                        overall.check()
                        run_guard.check()

                        def cancelled():
                            overall.check()
                            run_guard.check()
                            return False

                        def on_token(phase, _token):
                            overall.check()
                            run_guard.check()
                            key = f"{phase}_output_tokens"
                            progress = active_row["progress"]
                            progress[key] = progress.get(key, 0) + 1
                            if progress[key] == 1 or progress[key] % 32 == 0:
                                write_json(
                                    args.output / f"{run_id}.measurement.json",
                                    active_row,
                                )

                        song = pipe(**value, cancelled=cancelled, on_token=on_token)
                        overall.check()
                        run_guard.check()
                        frames = len(song.semantic.tokens)
                        if (
                            song.audio.shape != (1920 * frames - 64, 2)
                            or not np.isfinite(song.audio).all()
                        ):
                            raise AssertionError("Invalid decoded audio")
                        if args.workload != "modes" and (
                            song.truncated["abc"] or song.truncated["semantic"]
                        ):
                            raise AssertionError(
                                "Full-song acceptance cannot pass with truncated generation"
                            )
                        duration = len(song.audio) / 48000
                        active_row.update(
                            status="rendered_artifacts_pending",
                            audio_seconds=duration,
                            semantic_frames=frames,
                            truncated=song.truncated,
                            timing=song.timing,
                            listening_status="not_evaluated",
                        )
                        write_json(args.output / f"{run_id}.measurement.json", active_row)
                        write_json(args.output / "report.json", report)
                        song.save_artifacts(args.output / run_id)
                        overall.check()
                        run_guard.check()
                        active_row["status"] = "complete_listening_not_evaluated"
                        del song
                        gc.collect()
                        releases.append({"id": run_id, **memory_snapshot()})
                        overall.check()
                        run_guard.check()
                    active_row["resources"] = run_guard.report()
                    active_row["resource_evidence"] = _resource_evidence(
                        args.output, f"{run_id}.resources"
                    )
                    write_json(args.output / f"{run_id}.measurement.json", active_row)
                    write_json(args.output / "report.json", report)
                    print(
                        json.dumps(
                            {
                                key: active_row[key]
                                for key in ("id", "audio_seconds", "truncated", "timing")
                            }
                        ),
                        flush=True,
                    )
                    active_row = None
            overall.check()
        report["overall_resources"] = overall.report()
    except BaseException as exc:
        if active_row is not None:
            active_row["status"] = "failed_incomplete"
            active_row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            active_row["resource_evidence"] = _resource_evidence(
                args.output, f"{active_row['id']}.resources"
            )
            write_json(args.output / f"{active_row['id']}.measurement.json", active_row)
        report.update(
            status="failed_incomplete",
            construction_seconds=construction_seconds,
            error={"type": type(exc).__name__, "message": str(exc)},
            resource_evidence=_resource_evidence(args.output),
        )
        if overall is not None:
            report["overall_resources"] = overall.report()
        write_json(args.output / "report.json", report)
        raise

    report["resource_evidence"] = _resource_evidence(args.output)
    if args.workload == "modes":
        report["full_duration_gate"] = None
        report["full_duration_gate_status"] = "not_applicable_to_short_mode_fixtures"
    else:
        report["full_duration_gate"] = any(
            180 <= row["audio_seconds"] <= 240 for row in reports
        )
        report["full_duration_gate_status"] = (
            "passed" if report["full_duration_gate"] else "failed"
        )
    report["status"] = (
        "rendered_listening_not_evaluated"
        if report["full_duration_gate"] is not False
        else "failed_full_duration_gate"
    )
    write_json(args.output / "report.json", report)
    if args.workload != "modes" and not report["full_duration_gate"]:
        raise SystemExit("Rendered songs do not meet the recorded 3–4 minute acceptance workload")


if __name__ == "__main__":
    main()
