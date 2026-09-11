"""Compare MLX execution with saved real-checkpoint oracle tensors.

Measurement mode records errors without claiming acceptance. --limits enforces
separately justified tolerances; the report records which limits were applied.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import time

import mlx.core as mx
import numpy as np
import torch

from yue2.nar import Chunk
from yue2.storage import identity, sha256_file, write_json

from lyra.ar import load_ar, make_cache
from lyra.measure import GPUExecution
from lyra.nar import CachedNAR, load_nar

MAX_NAR_FRAMES = 500
PREFILL_CHUNK_SIZE = 1024
NAR_QUERY_CHUNK_SIZE = 256
SOLVER_STEPS = 32
_REQUIRED_GROUPS = {
    "ar": ("cache", "logits"),
    "nar": ("cache", "velocity", "state", "latents"),
    "vae": ("audio",),
}
_HIGHER_IS_BETTER = frozenset({"cosine", "argmax_agreement"})


def _atomic_savez(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resource_evidence(output):
    result = {}
    for name in ("resources.jsonl", "resources.json"):
        path = Path(output) / name
        if path.is_file():
            result[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _guard_cancelled(guard):
    guard.check()
    return False


def _invocation(args):
    hashes = {"oracle_npz": sha256_file(args.oracle / f"{args.stage}.npz")}
    candidates = {
        "oracle_metadata": args.oracle / f"{args.stage}.json",
        "conversion": Path(args.model) / "conversion.json",
        "model_config": Path(args.model) / "config.json",
        "limits": args.limits,
        "latents": args.latents,
    }
    for name, path in candidates.items():
        if path is not None and Path(path).is_file():
            hashes[name] = sha256_file(path)
    record = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "script_sha256": sha256_file(__file__),
        "source_sha256": {
            path.name: sha256_file(path)
            for path in sorted((Path(__file__).resolve().parents[1] / "src" / "lyra").glob("*.py"))
        },
        "input_sha256": hashes,
        "runtime": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            **{
                name: version(name)
                for name in ("mlx", "mlx-lm", "torch", "transformers", "numpy")
            },
        },
        "execution_config": {
            "backend": "mps" if args.stage == "vae" else "mlx",
            "memory_budget_gib": args.memory_budget_gib,
            "ar_prefill_chunk_size": args.prefill_chunk_size if args.stage == "ar" else None,
            "selected_ar_lengths": args.lengths if args.stage == "ar" else None,
            "max_nar_frames": MAX_NAR_FRAMES if args.stage == "nar" else None,
            "nar_solver_steps": SOLVER_STEPS if args.stage == "nar" else None,
            "nar_query_chunk_size": (
                args.query_chunk_size if args.stage == "nar" else None
            ),
            "dtypes": {
                "ar_weights": args.precision if args.stage == "ar" else None,
                "ar_cache": "bfloat16" if args.stage == "ar" else None,
                "nar_conditioning": "bfloat16" if args.stage == "nar" else None,
                "nar_solver": "bfloat16" if args.stage == "nar" else None,
                "vae": "float32" if args.stage == "vae" else None,
            },
        },
    }
    record["invocation_sha256"] = identity(record)
    return record


def metrics(expected, actual):
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {expected.shape} vs {actual.shape}")
    if expected.size == 0:
        raise ValueError("Fidelity tensors must be nonempty")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise FloatingPointError("Non-finite fidelity tensor")
    delta = actual - expected
    rms = float(np.sqrt(np.mean(expected * expected)))
    error = float(np.sqrt(np.mean(delta * delta)))
    denominator = float(np.linalg.norm(expected.ravel()) * np.linalg.norm(actual.ravel()))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms_error": error,
        "reference_rms": rms,
        "relative_rms_error": error / max(rms, 1e-12),
        "cosine": (
            float(np.vdot(expected.ravel(), actual.ravel()) / denominator)
            if denominator
            else float(np.array_equal(expected, actual))
        ),
        "p99_abs": float(np.quantile(np.abs(delta), 0.99)),
    }


def as_numpy(value):
    return np.asarray(value.astype(mx.float32))


def compare_ar(args, guard, persist):
    guard.check()
    model = load_ar(args.model, precision=args.precision)
    mx.eval(model.parameters())
    guard.check()
    report, actual = {}, {}
    partial = args.output / "ar-actual.partial.npz"
    try:
        with np.load(args.oracle / "ar.npz", allow_pickle=False) as reference:
            available = sorted(
                int(name.split("_")[1])
                for name in reference.files
                if name.startswith("ids_")
            )
            if not available:
                raise ValueError("AR oracle contains no teacher-forced cases")
            lengths = list(args.lengths) if args.lengths is not None else available
            missing = [length for length in lengths if length not in available]
            if missing:
                raise ValueError(f"Requested AR lengths are absent from oracle: {missing}")
            completed = []
            for length in lengths:
                guard.check()
                ids = reference[f"ids_{length}"].tolist()
                if len(ids) != length or length < 5:
                    raise ValueError(f"Invalid AR oracle case length {length}")
                cache = make_cache(model, length)
                start_time = time.perf_counter()
                try:
                    inputs = mx.array([ids], dtype=mx.int32)
                    prefill_count = length - 4
                    hidden = None
                    for start in range(0, prefill_count, args.prefill_chunk_size):
                        guard.check()
                        stop = min(start + args.prefill_chunk_size, prefill_count)
                        hidden = model.model(inputs[:, start:stop], cache=cache)
                        mx.eval(hidden, [item.state for item in cache])
                        guard.check()
                        if stop < prefill_count:
                            del hidden
                            hidden = None
                            mx.clear_cache()
                    if hidden is None:
                        raise RuntimeError("AR fixture requires a nonempty prefill")
                    logits = [as_numpy(model.lm_head(hidden[:, -1:])[0, -1])]
                    guard.check()
                    for token in ids[-4:]:
                        guard.check()
                        hidden = model.model(mx.array([[token]], dtype=mx.int32), cache=cache)
                        mx.eval(hidden, [item.state for item in cache])
                        logits.append(as_numpy(model.lm_head(hidden)[0, -1]))
                        guard.check()
                    array = np.stack(logits)
                    actual[f"logits_{length}"] = array
                    report[f"logits_{length}"] = metrics(reference[f"logits_{length}"], array)
                    report[f"logits_{length}"]["argmax_agreement"] = float(
                        np.mean(array.argmax(-1) == reference[f"logits_{length}"].argmax(-1))
                    )
                    positions = mx.array(reference[f"positions_{length}"], dtype=mx.int32)
                    for layer, item in enumerate(cache):
                        for kind, data in (("k", item.keys), ("v", item.values)):
                            if data.dtype != mx.bfloat16:
                                raise TypeError("AR cache must remain BF16 at every position")
                            name = f"{kind}_{length}_{layer}"
                            if name not in reference.files:
                                raise KeyError(f"Oracle is missing required cache tensor {name}")
                            actual[name] = as_numpy(mx.take(data, positions, axis=2))
                            report[name] = metrics(reference[name], actual[name])
                    mx.synchronize()
                    guard.check()
                    report[f"logits_{length}"]["seconds"] = time.perf_counter() - start_time
                    completed.append(length)
                    _atomic_savez(partial, actual)
                    persist(
                        report,
                        {
                            "completed_lengths": completed,
                            "selected_lengths": lengths,
                            "prefill_chunk_size": args.prefill_chunk_size,
                            "active_length": None,
                        },
                    )
                    print(json.dumps({"case": length, **report[f"logits_{length}"]}), flush=True)
                finally:
                    for item in cache:
                        item.close()
                    del cache
                    mx.clear_cache()
        _atomic_savez(args.output / "ar-actual.npz", actual)
        partial.unlink(missing_ok=True)
        return report, {
            "lengths": lengths,
            "prefill_chunk_size": args.prefill_chunk_size,
            "position_ids": "absolute_cache_offsets",
            "key_visibility": "full_causal_history",
            "weight_precision": args.precision,
            "cache_dtype": "bfloat16",
        }
    finally:
        del model
        mx.clear_cache()


def compare_nar(args, guard, persist):
    metadata_path = args.oracle / "nar.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("NAR oracle metadata nar.json is required")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("steps") != SOLVER_STEPS:
        raise ValueError(f"NAR fidelity requires the original {SOLVER_STEPS}-step midpoint oracle")
    guard.check()
    ar = load_ar(args.model, precision="bf16")
    model = load_nar(args.model, ar)
    mx.eval(model.parameters())
    guard.check()
    report, actual = {}, {}
    engine = None
    partial = args.output / "nar-actual.partial.npz"
    try:
        with np.load(args.oracle / "nar.npz", allow_pickle=False) as reference:
            codec = reference["codec"].tolist()
            prefix = reference["prefix"].tolist()
            if not 1 <= len(codec) <= MAX_NAR_FRAMES:
                raise ValueError(f"NAR fidelity oracle must contain 1..{MAX_NAR_FRAMES} frames")
            if metadata.get("frames") != len(codec):
                raise ValueError("NAR oracle metadata frame count disagrees with nar.npz")
            from yue2.protocol import CODEC_OFFSET, MUSIC_END

            chunk = Chunk(
                prefix + [token + CODEC_OFFSET for token in codec] + [MUSIC_END],
                torch.from_numpy(reference["noise"].copy()),
            )
            engine = CachedNAR(
                model,
                chunk,
                query_chunk_size=args.query_chunk_size,
                cancelled=lambda: _guard_cancelled(guard),
            )
            guard.check()
            for layer, (key, value) in enumerate(engine.cache):
                for kind, data in (("k", key), ("v", value)):
                    if data.dtype != mx.bfloat16:
                        raise TypeError("NAR conditioning cache must remain BF16")
                    name = f"{kind}_{layer}"
                    if name not in reference.files:
                        raise KeyError(f"Oracle is missing required cache tensor {name}")
                    actual[name] = as_numpy(data)
                    report[name] = metrics(reference[name], actual[name])
            persist(
                report,
                {
                    "frames": len(codec),
                    "solver_steps": SOLVER_STEPS,
                    "completed_midpoint_steps": 0,
                    "production_solve_complete": False,
                },
            )

            # Hold the reference state fixed to isolate velocity evaluation from
            # accumulated trajectory differences in the separate solver replay.
            for step in (0, 1, 15, 31):
                for name, state_name, t in (
                    (f"velocity_{step}", f"state_{step}", 1.0 - step / SOLVER_STEPS),
                    (
                        f"mid_velocity_{step}",
                        f"mid_{step}",
                        1.0 - (step + 0.5) / SOLVER_STEPS,
                    ),
                ):
                    guard.check()
                    raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
                    fixed = mx.array(reference[state_name], dtype=mx.bfloat16)
                    key = f"fixed_{name}"
                    actual[key] = as_numpy(engine.velocity(fixed, raw))
                    report[key] = metrics(reference[name], actual[key])

            state = mx.array(reference["noise"], dtype=mx.bfloat16)
            dt = 1.0 / SOLVER_STEPS
            half_dt = mx.array(dt / 2.0, dtype=mx.float32)
            full_dt = mx.array(dt, dtype=mx.float32)
            mx.eval(state, half_dt, full_dt)
            for step in range(SOLVER_STEPS):
                state_name = f"state_{step}"
                if state_name in reference.files:
                    actual[state_name] = as_numpy(state)
                    report[state_name] = metrics(reference[state_name], actual[state_name])
                t = 1.0 - step * dt
                raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
                guard.check()
                first = engine.velocity(state, raw)
                mx.eval(first)
                guard.check()
                velocity_name = f"velocity_{step}"
                if velocity_name in reference.files:
                    actual[velocity_name] = as_numpy(first)
                    report[velocity_name] = metrics(reference[velocity_name], actual[velocity_name])
                midpoint = state - (first.astype(mx.float32) * half_dt).astype(mx.bfloat16)
                mx.eval(midpoint)
                mid_name = f"mid_{step}"
                if mid_name in reference.files:
                    actual[mid_name] = as_numpy(midpoint)
                    report[mid_name] = metrics(reference[mid_name], actual[mid_name])
                raw_mid = torch.logit(
                    torch.tensor(t - dt / 2.0, dtype=torch.float64)
                ).clamp(-20, 20).item()
                guard.check()
                velocity = engine.velocity(midpoint, raw_mid)
                mx.eval(velocity)
                guard.check()
                mid_velocity_name = f"mid_velocity_{step}"
                if mid_velocity_name in reference.files:
                    actual[mid_velocity_name] = as_numpy(velocity)
                    report[mid_velocity_name] = metrics(
                        reference[mid_velocity_name], actual[mid_velocity_name]
                    )
                state = state - (velocity.astype(mx.float32) * full_dt).astype(mx.bfloat16)
                mx.eval(state)
                guard.check()
                if any(
                    name in reference.files
                    for name in (state_name, velocity_name, mid_name, mid_velocity_name)
                ):
                    _atomic_savez(partial, actual)
                persist(
                    report,
                    {
                        "frames": len(codec),
                        "solver_steps": SOLVER_STEPS,
                        "completed_midpoint_steps": step + 1,
                        "production_solve_complete": False,
                    },
                )

            actual["reproduced_latents"] = as_numpy(state)
            report["reproduced_latents"] = metrics(
                reference["latents"], actual["reproduced_latents"]
            )
            _atomic_savez(partial, actual)
            persist(
                report,
                {
                    "frames": len(codec),
                    "solver_steps": SOLVER_STEPS,
                    "completed_midpoint_steps": SOLVER_STEPS,
                    "production_solve_complete": False,
                },
            )
            guard.check()
            production_evaluations = 0
            source_velocity = engine.velocity

            def counted_velocity(state, raw_t):
                nonlocal production_evaluations
                if state.dtype != mx.bfloat16:
                    raise TypeError("Production solver states must remain BF16")
                result = source_velocity(state, raw_t)
                if result.dtype != mx.bfloat16:
                    raise TypeError("Production velocity outputs must remain BF16")
                production_evaluations += 1
                return result

            engine.velocity = counted_velocity
            try:
                actual["latents"] = np.asarray(
                    engine.solve(
                        steps=SOLVER_STEPS,
                        cancelled=lambda: _guard_cancelled(guard),
                        on_progress=lambda _completed, _total: guard.check(),
                    ),
                    dtype=np.float32,
                )
            finally:
                del engine.velocity
            if production_evaluations != 2 * SOLVER_STEPS:
                raise ValueError("Midpoint integration must evaluate two velocities per step")
            guard.check()
            report["latents"] = metrics(reference["latents"], actual["latents"])
            _atomic_savez(args.output / "nar-actual.npz", actual)
            partial.unlink(missing_ok=True)
            persist(
                report,
                {
                    "frames": len(codec),
                    "solver_steps": SOLVER_STEPS,
                    "completed_midpoint_steps": SOLVER_STEPS,
                    "production_solve_complete": True,
                },
            )
            print(
                json.dumps(
                    {
                        "latents": report["latents"],
                        "reproduced_latents": report["reproduced_latents"],
                        "velocity_0": report.get("velocity_0"),
                    }
                ),
                flush=True,
            )
            return report, {
                "frames": len(codec),
                "solver_steps": SOLVER_STEPS,
                "state_transition": "production_bf16_midpoint_arithmetic",
                "production_solve_complete": True,
                "production_velocity_evaluations": production_evaluations,
                "conditioning_dtype": "bfloat16",
                "solver_dtype": "bfloat16",
                "query_chunk_size": args.query_chunk_size,
                "key_visibility": "full_conditioning_and_acoustic_state",
            }
    finally:
        if engine is not None:
            engine.close()
        del model, ar
        mx.clear_cache()


def compare_vae(args, guard, persist):
    from yue2.modeling_vae import YuE2VAE

    guard.check()
    model = YuE2VAE.from_pretrained(
        args.vae,
        decoder_only=True,
        device="mps",
        local_files_only=True,
    )
    try:
        if args.latents.endswith(".npz"):
            with np.load(args.latents, allow_pickle=False) as data:
                latents = data["latents"].copy()
        else:
            latents = np.load(args.latents, allow_pickle=False)
        z = torch.from_numpy(latents.T[None])
        report, actual_outputs = {}, {}
        partial = args.output / "vae-actual.partial.npz"
        with np.load(args.oracle / "vae.npz", allow_pickle=False) as reference:
            completed = []
            if not reference.files:
                raise ValueError("VAE oracle contains no decoded outputs")
            cases = list(reference.files)
            if "core_256" not in cases:
                cases.append("core_256")
            for key in cases:
                guard.check()
                if key == "full":
                    actual = model.decode(z).cpu().numpy()
                else:
                    core = int(key.split("_")[1])
                    actual = model.decode_tiled(
                        z,
                        core_frames=core,
                        halo_frames=16,
                        on_progress=lambda _completed, _total: guard.check(),
                    ).numpy()
                guard.check()
                expected = reference[key] if key in reference.files else reference["full"]
                actual_outputs[key] = actual
                report[key] = metrics(expected, actual)
                _atomic_savez(partial, actual_outputs)
                completed.append(key)
                persist(report, {"completed_outputs": completed})
        _atomic_savez(args.output / "vae-actual.npz", actual_outputs)
        partial.unlink(missing_ok=True)
        return report, {
            "outputs": completed,
            "frames": z.shape[-1],
            "decoder_dtype": "float32",
        }
    finally:
        del model
        torch.mps.empty_cache()


def _metric_group(name):
    if name.startswith(("k_", "v_")):
        return "cache"
    if name.startswith("logits_"):
        return "logits"
    if "velocity" in name:
        return "velocity"
    if name.startswith(("state_", "mid_")):
        return "state"
    if name == "latents" or name.endswith("_latents"):
        return "latents"
    return "audio"


def _bounds(metric, specification):
    if isinstance(specification, bool):
        raise TypeError("boolean tolerance bounds are invalid")
    if isinstance(specification, (int, float)):
        value = float(specification)
        if not math.isfinite(value):
            raise ValueError("tolerance bounds must be finite")
        return ({"lower": value} if metric in _HIGHER_IS_BETTER else {"upper": value})
    if not isinstance(specification, dict) or not specification:
        raise TypeError("a tolerance must be a number or a nonempty lower/upper object")
    unknown = set(specification) - {"lower", "upper"}
    if unknown:
        raise ValueError(f"unknown tolerance bound keys: {sorted(unknown)}")
    result = {}
    for direction, value in specification.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("tolerance bounds must be finite numbers")
        result[direction] = float(value)
    if result.get("lower", -math.inf) > result.get("upper", math.inf):
        raise ValueError("tolerance lower bound exceeds upper bound")
    return result


def apply_limits(stage, measured, limits):
    failures = []
    normalized = {"tensors": {}}
    stage_limits = limits.get(stage) if isinstance(limits, dict) else None
    if not isinstance(stage_limits, dict):
        return [
            {
                "kind": "limits",
                "reason": f"limits are missing required stage {stage!r}",
            }
        ], normalized

    required = set(_REQUIRED_GROUPS[stage])
    missing_groups = sorted(required - set(stage_limits))
    extra_groups = sorted(set(stage_limits) - required - {"tensors"})
    failures.extend(
        {"kind": "limits", "group": group, "reason": "missing required limit group"}
        for group in missing_groups
    )
    failures.extend(
        {"kind": "limits", "group": group, "reason": "unknown limit group"}
        for group in extra_groups
    )

    grouped = {group: [] for group in required}
    tensor_groups = {}
    for name, values in measured.items():
        group = _metric_group(name)
        tensor_groups[name] = group
        if group in grouped:
            grouped[group].append((name, values))
        else:
            failures.append(
                {
                    "kind": "measurement",
                    "tensor": name,
                    "reason": f"unmapped metric group {group!r}",
                }
            )

    group_bounds = {}
    for group in sorted(required):
        tensors = grouped[group]
        if not tensors:
            failures.append(
                {
                    "kind": "measurement",
                    "group": group,
                    "reason": "required metric group has no tensors",
                }
            )
        specifications = stage_limits.get(group)
        if not isinstance(specifications, dict) or not specifications:
            failures.append(
                {
                    "kind": "limits",
                    "group": group,
                    "reason": "required limit group has no metrics",
                }
            )
            continue
        normalized[group] = {}
        group_bounds[group] = {}
        for metric, specification in specifications.items():
            try:
                bounds = _bounds(metric, specification)
            except (TypeError, ValueError) as exc:
                failures.append(
                    {
                        "kind": "limits",
                        "group": group,
                        "metric": metric,
                        "reason": str(exc),
                    }
                )
                continue
            normalized[group][metric] = bounds
            group_bounds[group][metric] = bounds

    overrides = stage_limits.get("tensors", {})
    if not isinstance(overrides, dict):
        failures.append(
            {
                "kind": "limits",
                "group": "tensors",
                "reason": "per-tensor overrides must be an object",
            }
        )
        overrides = {}
    tensor_bounds = {}
    for name, specifications in overrides.items():
        if name not in measured:
            failures.append(
                {
                    "kind": "limits",
                    "tensor": name,
                    "reason": "per-tensor override names an unavailable tensor",
                }
            )
            continue
        group = tensor_groups[name]
        if not isinstance(specifications, dict) or not specifications:
            failures.append(
                {
                    "kind": "limits",
                    "tensor": name,
                    "group": group,
                    "reason": "per-tensor override has no metrics",
                }
            )
            continue
        normalized["tensors"][name] = {}
        tensor_bounds[name] = {}
        for metric, specification in specifications.items():
            try:
                bounds = _bounds(metric, specification)
            except (TypeError, ValueError) as exc:
                failures.append(
                    {
                        "kind": "limits",
                        "tensor": name,
                        "group": group,
                        "metric": metric,
                        "reason": str(exc),
                    }
                )
                continue
            normalized["tensors"][name][metric] = bounds
            tensor_bounds[name][metric] = bounds

    for group, tensors in grouped.items():
        for name, values in tensors:
            effective = {
                **group_bounds.get(group, {}),
                **tensor_bounds.get(name, {}),
            }
            for metric, bounds in effective.items():
                if metric not in values:
                    failures.append(
                        {
                            "kind": "measurement",
                            "tensor": name,
                            "group": group,
                            "metric": metric,
                            "reason": "required metric is missing",
                        }
                    )
                    continue
                actual = values[metric]
                if "lower" in bounds and actual < bounds["lower"]:
                    failures.append(
                        {
                            "kind": "bound",
                            "tensor": name,
                            "group": group,
                            "metric": metric,
                            "actual": actual,
                            "lower": bounds["lower"],
                            "source": (
                                "tensor"
                                if metric in tensor_bounds.get(name, {})
                                else "group"
                            ),
                        }
                    )
                if "upper" in bounds and actual > bounds["upper"]:
                    failures.append(
                        {
                            "kind": "bound",
                            "tensor": name,
                            "group": group,
                            "metric": metric,
                            "actual": actual,
                            "upper": bounds["upper"],
                            "source": (
                                "tensor"
                                if metric in tensor_bounds.get(name, {})
                                else "group"
                            ),
                        }
                    )
    return failures, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ar", "nar", "vae"))
    parser.add_argument("--model", default="models/converted")
    parser.add_argument("--precision", choices=("bf16", "8bit", "4bit"), default="bf16")
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limits", type=Path)
    parser.add_argument("--vae")
    parser.add_argument("--latents")
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--prefill-chunk-size", type=int, default=PREFILL_CHUNK_SIZE)
    parser.add_argument("--query-chunk-size", type=int, default=NAR_QUERY_CHUNK_SIZE)
    parser.add_argument("--memory-budget-gib", type=float, default=16)
    parser.add_argument("--require-ac", action="store_true")
    args = parser.parse_args()
    if args.stage != "ar" and args.precision != "bf16":
        parser.error("--precision selects AR weights; NAR is BF16 and VAE is FP32")
    precision = "float32" if args.stage == "vae" else args.precision

    if args.stage == "ar" and args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be a positive integer")
    if args.stage == "nar" and args.query_chunk_size < 1:
        parser.error("--query-chunk-size must be a positive integer")
    if args.lengths is not None and (
        len(set(args.lengths)) != len(args.lengths) or any(length < 5 for length in args.lengths)
    ):
        parser.error("--lengths must contain unique integers of at least 5")
    if args.stage != "ar" and args.lengths is not None:
        parser.error("--lengths is only valid for AR fidelity")
    if args.stage == "vae" and (args.vae is None or args.latents is None):
        parser.error("VAE fidelity requires --vae and --latents")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Fidelity capture requires an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    invocation = _invocation(args)
    write_json(args.output / "invocation.json", invocation)
    progress = {"metrics": {}, "execution": {}}
    partial_path = args.output / "fidelity.partial.json"

    def persist(metrics_value, execution):
        progress["metrics"] = metrics_value
        progress["execution"] = execution
        write_json(
            partial_path,
            {
                "status": "running",
                "stage": args.stage,
                "precision": precision,
                "invocation_sha256": invocation["invocation_sha256"],
                "execution": execution,
                "metrics": metrics_value,
            },
        )

    write_json(
        partial_path,
        {
            "status": "running",
            "stage": args.stage,
            "precision": precision,
            "invocation_sha256": invocation["invocation_sha256"],
            "execution": {},
            "metrics": {},
        },
    )
    backend = "mps" if args.stage == "vae" else "mlx"
    base_report = {
        "stage": args.stage,
        "precision": precision,
        "invocation_sha256": invocation["invocation_sha256"],
        "oracle_sha256": invocation["input_sha256"]["oracle_npz"],
        "conversion_sha256": invocation["input_sha256"].get("conversion"),
        "limits": None,
        "status": "measured_not_accepted",
    }
    try:
        with GPUExecution(
            backend=backend,
            memory_budget_gib=args.memory_budget_gib,
            resource_path=args.output / "resources.jsonl",
            require_ac=args.require_ac,
        ) as guard:
            guard.check()
            measured, execution = {
                "ar": compare_ar,
                "nar": compare_nar,
                "vae": compare_vae,
            }[args.stage](args, guard, persist)
            guard.check()
    except BaseException as exc:
        report = {
            **base_report,
            "status": "failed_incomplete",
            "execution": progress["execution"],
            "metrics": progress["metrics"],
            "failures": [
                {"kind": "execution", "type": type(exc).__name__, "message": str(exc)}
            ],
            "resources": _resource_evidence(args.output),
        }
        write_json(args.output / "fidelity.json", report)
        raise

    report = {**base_report, "execution": execution, "metrics": measured}
    actual_path = args.output / f"{args.stage}-actual.npz"
    report["actual_sha256"] = sha256_file(actual_path)
    failures = []
    if args.limits:
        try:
            serialized_limits = args.limits.read_text()
            limits_sha256 = sha256_file(args.limits)
            limits = json.loads(serialized_limits)
            failures, normalized = apply_limits(args.stage, measured, limits)
            calibration = limits.get("calibration") if isinstance(limits, dict) else None
            if calibration is not None and (
                not isinstance(calibration, dict)
                or calibration.get("reference_npz_sha256") != base_report["oracle_sha256"]
            ):
                failures.append({"kind": "limits", "reason": "calibration belongs to another reference"})
            if limits_sha256 != invocation["input_sha256"].get("limits"):
                failures.append(
                    {
                        "kind": "limits",
                        "reason": "limit file changed after invocation was recorded",
                    }
                )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            limits_sha256 = None
            limits = None
            normalized = {}
            failures = [
                {
                    "kind": "limits",
                    "reason": f"could not read valid limit JSON: {type(exc).__name__}: {exc}",
                }
            ]
        report["limits"] = {
            "sha256": limits_sha256,
            "values": limits,
            "normalized_stage": normalized,
            "required_groups": list(_REQUIRED_GROUPS[args.stage]),
        }
        report["status"] = "failed" if failures else "passed"
    report["failures"] = failures
    report["resources"] = _resource_evidence(args.output)
    write_json(args.output / "fidelity.json", report)
    partial_path.unlink(missing_ok=True)
    if failures:
        raise SystemExit(f"Fidelity comparison failed closed: {len(failures)} violations")


if __name__ == "__main__":
    main()
