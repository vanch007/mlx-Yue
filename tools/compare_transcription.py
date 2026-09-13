"""Compare identity-bound complete token trajectories, separately from float metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from fidelity import metrics
from yue2.storage import sha256_file, write_json


def compare(reference, actual):
    reports = [json.loads((path / "report.json").read_text()) for path in (reference, actual)]
    for path, report in zip((reference, actual), reports):
        if report["values_sha256"] != sha256_file(path / "values.npz"):
            raise ValueError("Transcription capture hash changed")
    for key in ("input_sha256", "weights_sha256", "task", "complete_generation"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Transcription captures disagree: {key}")
    if not reports[0]["complete_generation"] or [r["backend"] for r in reports] != ["torch", "mlx"]:
        raise ValueError("Expected complete original Torch and MLX generation captures")
    with np.load(reference / "values.npz") as ref, np.load(actual / "values.npz") as port:
        a, b = ref["generated"], port["generated"]
        if a.dtype.kind not in "iu" or b.dtype != a.dtype or a.ndim != 1 or b.ndim != 1:
            raise ValueError("Expected matching one-dimensional integer token dtypes")
        equal = np.array_equal(a, b)
        common = min(len(a), len(b))
        mismatches = np.flatnonzero(a[:common] != b[:common])
        first = int(mismatches[0]) if len(mismatches) else common if len(a) != len(b) else None
        measured = {k: metrics(ref[k], port[k]) for k in ("memory", "logits", "cached")}
    return {"status": "pass" if equal else "fail", "scope": "Exact complete greedy token trajectory on the recorded input/task only",
            "reference_tokens": len(a), "actual_tokens": len(b), "first_mismatch": first,
            "token_dtype": str(a.dtype), "float_metrics": measured,
            "float_numerical_acceptance": "measured_not_accepted", "identities": reports,
            "reference_report_sha256": sha256_file(reference / "report.json"),
            "actual_report_sha256": sha256_file(actual / "report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.reference, args.actual)
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "reference_tokens", "actual_tokens", "first_mismatch")}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
