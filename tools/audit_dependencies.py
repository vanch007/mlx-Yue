"""Audit one frozen installation profile against the live PyPI advisory database.

Run with Python 3.12 on the target platform. No project packages are installed;
uv runs the pinned auditor in its own environment. JSON retains skipped packages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("runtime", "transcription", "dev"),
                        default="runtime")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    requirements = args.output.resolve() / f"{args.profile}-requirements.txt"
    report = args.output.resolve() / f"{args.profile}-audit.json"
    command = ["uv", "export", "--locked", "--no-emit-project", "--no-default-groups"]
    if args.profile in {"transcription", "dev"}:
        command += ["--extra", "transcription"]
    if args.profile == "dev":
        command += ["--group", "dev"]
    subprocess.run([*command, "--output-file", str(requirements)], cwd=root,
                   stdout=subprocess.DEVNULL, check=True)
    result = subprocess.run([
        "uvx", "--python", "3.12", "pip-audit==2.10.1", "--disable-pip", "--no-deps",
        "-r", str(requirements), "--format", "json", "--output", str(report),
    ], cwd=root)
    if result.returncode:
        return result.returncode
    data = json.loads(report.read_text())
    skipped = [d for d in data["dependencies"] if d.get("skip_reason")]
    if skipped:
        print(f"Incomplete audit coverage: {skipped}", file=sys.stderr)
        return 2
    print(f"{args.profile}: {len(data['dependencies'])} packages; no known advisories matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
