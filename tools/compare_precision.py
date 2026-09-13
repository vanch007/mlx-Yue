"""Serial, isolated-process BF16/8bit generation comparison with retained evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, request in enumerate(args.requests):
        # Reverse order on alternating requests to expose ordering effects.
        precisions = ("bf16", "8bit") if index % 2 == 0 else ("8bit", "bf16")
        for precision in precisions:
            name = request.stem + "-" + precision
            destination = args.output / name
            command = [sys.executable, "-m", "lyra.cli", "generate", str(request.resolve()),
                       "--model", str(args.model.resolve()), "--vae", str(args.vae.resolve()),
                       "--precision", precision, "--offline", "--output", str(destination), "--require-ac"]
            start = time.perf_counter()
            print(json.dumps({"starting": name}), flush=True)
            with (args.output / (name + ".log")).open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((destination / "result.json").read_text())
            samples = [json.loads(line) for line in destination.with_name(name + ".resources.jsonl").read_text().splitlines()]
            record = {
                "case": request.stem, "precision": precision, "directory": str(destination.resolve()),
                "wall_seconds": time.perf_counter() - start, "audio_seconds": result["audio_seconds"],
                "timing": result["timing"], "truncated": result["truncated"],
                "peak_footprint_gib": max(s["physical_footprint_bytes"] for s in samples) / 2**30,
                "identity": result["identity"],
            }
            records.append(record)
            (args.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    subprocess.run([sys.executable, "-m", "lyra.cli", "listen", *[r["directory"] for r in records],
                    "--output", str(args.output / "listen")], check=False)


if __name__ == "__main__":
    main()
