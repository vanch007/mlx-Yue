"""Derive BF16 comparison budgets from independent FP32 reference captures.

Two BF16 implementations each within the reference's measured FP32 error radius
can differ by at most twice that radius (triangle inequality for RMS and L-infinity).
This is an explicit acceptance budget, not a claim that all BF16 errors obey it.
No MLX output is read when selecting the limits. Per-tensor limits prevent a large
late-layer error budget from hiding an early-layer discrepancy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fidelity import _metric_group, apply_limits, metrics
from yue2.storage import sha256_file, write_json


def derive(stage, reference_dir, calibration_dir):
    reference_path = reference_dir / f"{stage}.npz"
    calibration_path = calibration_dir / f"{stage}.npz"
    limits, radii = {"tensors": {}}, {}
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        calibration_path, allow_pickle=False
    ) as anchor:
        if stage == "vae":
            if "full" not in reference.files or "core_64" not in reference.files:
                raise ValueError("VAE calibration needs full and 64-frame tiled FP32 audio")
            values = metrics(reference["full"], reference["core_64"])
            radii["upstream_fp32_tiling"] = values
            limits["audio"] = {key: 2 * values[key] for key in ("rms_error", "max_abs")}
        else:
            expected_identity = json.loads((reference_dir / f"{stage}.json").read_text())["model"]
            captured_identity = json.loads((calibration_dir / "invocation.json").read_text())["weights"]
            if expected_identity != captured_identity:
                raise ValueError("BF16 and FP32 captures must use identical checkpoint content")
            for name in anchor.files:
                if name.startswith(("ids_", "positions_")) or name in {"prefix", "codec", "noise"}:
                    if (
                        name not in reference.files
                        or anchor[name].dtype != reference[name].dtype
                        or not np.array_equal(anchor[name], reference[name])
                    ):
                        raise ValueError(f"BF16 and FP32 capture inputs differ: {name}")
                    continue
                source_name = name.removeprefix("fixed_")
                if source_name not in reference.files:
                    raise ValueError(f"FP32 tensor has no BF16 reference: {name}")
                values = metrics(anchor[name], reference[source_name])
                radii[name] = values
                bounds = {key: 2 * values[key] for key in ("rms_error", "max_abs")}
                limits["tensors"][name] = bounds
                group = limits.setdefault(_metric_group(name), {})
                for metric, bound in bounds.items():
                    group[metric] = max(group.get(metric, 0), bound)
            if stage == "nar":
                limits["tensors"]["reproduced_latents"] = dict(limits["tensors"]["latents"])
    return {
        stage: limits,
        "calibration": {
            "method": "two_equal_reference_precision_error_radii",
            "reference_npz_sha256": sha256_file(reference_path),
            "anchor_npz_sha256": sha256_file(calibration_path),
            "anchor_invocation_sha256": (
                sha256_file(calibration_dir / "invocation.json")
                if (calibration_dir / "invocation.json").is_file() else None
            ),
            "reference_error_radii": radii,
            "uses_port_outputs": False,
            "metric_rationale": "RMS and maximum absolute error satisfy the triangle inequality",
            "scope": "FP32 tiling sensitivity" if stage == "vae" else "BF16 versus FP32 arithmetic",
        },
    }


def check_actual(stage, reference_dir, calibration_dir, actual_path, limits_path):
    limits = json.loads(limits_path.read_text())
    anchor_path = calibration_dir / f"{stage}.npz"
    reference_path = reference_dir / f"{stage}.npz"
    provenance = limits["calibration"]
    if provenance["anchor_npz_sha256"] != sha256_file(anchor_path):
        raise ValueError("Acceptance limits belong to another FP32 anchor")
    if provenance["reference_npz_sha256"] != sha256_file(reference_path):
        raise ValueError("Acceptance limits belong to another BF16 reference")
    comparison_path = actual_path.parent / "fidelity.json"
    comparison = json.loads(comparison_path.read_text())
    actual_hash = sha256_file(actual_path)
    if comparison["oracle_sha256"] != provenance["reference_npz_sha256"]:
        raise ValueError("Actual tensors were compared against another reference")
    if comparison.get("actual_sha256", actual_hash) != actual_hash:
        raise ValueError("Actual tensors changed after their fidelity comparison")
    measured = {}
    with np.load(anchor_path, allow_pickle=False) as anchor, np.load(
        actual_path, allow_pickle=False
    ) as actual:
        for name in actual.files:
            anchor_name = "full" if stage == "vae" else (
                "latents" if name == "reproduced_latents" else name
            )
            measured[name] = metrics(anchor[anchor_name], actual[name])
            if name.startswith("logits_"):
                measured[name]["argmax_agreement"] = float(np.mean(
                    anchor[anchor_name].argmax(-1) == actual[name].argmax(-1)
                ))
    failures, _ = apply_limits(stage, measured, limits)
    if comparison["status"] != "passed":
        failures.append({"kind": "reference_comparison", "status": comparison["status"]})
    return {
        "stage": stage, "status": "failed" if failures else "passed",
        "comparison": "Port against the independent FP32 anchor, using the existing error budget",
        "reference_comparison_status": comparison["status"],
        "input_sha256": {
            "anchor": sha256_file(anchor_path), "actual": actual_hash,
            "limits": sha256_file(limits_path), "fidelity_report": sha256_file(comparison_path),
        },
        "metrics": measured, "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ar", "nar", "vae"))
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual", type=Path, help="Check saved port tensors against the FP32 anchor")
    parser.add_argument("--limits", type=Path, help="Previously derived, immutable error budget")
    args = parser.parse_args()
    if (args.actual is None) != (args.limits is None):
        parser.error("--actual and --limits must be provided together")
    if args.output.exists():
        raise FileExistsError("Derived acceptance limits are never silently overwritten")
    result = (derive(args.stage, args.oracle, args.calibration) if args.actual is None else
              check_actual(args.stage, args.oracle, args.calibration, args.actual, args.limits))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    if args.actual is not None:
        print(json.dumps({"stage": args.stage, "status": result["status"],
                          "tensor_count": len(result["metrics"]), "failures": result["failures"]}))
        if result["failures"]:
            raise SystemExit("Independent FP32 anchor comparison failed")
        return
    print(json.dumps({"stage": args.stage, "groups": {key: value for key, value in result[args.stage].items()
                                                    if key != "tensors"},
                      "per_tensor_limits": len(result[args.stage]["tensors"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
