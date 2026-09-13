"""Guarded complete-song reference using the pinned original Torch pipeline."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import torch

from lyra.measure import GPUExecution
from yue2.pipeline import YuE2Pipeline
from yue2.storage import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if version("torch") != "2.11.0" or version("transformers") != "4.57.6":
        raise RuntimeError("Use the pinned .oracle runtime")
    torch.set_float32_matmul_precision("highest")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    request = json.loads(args.request.read_text())
    report = {"status": "running", "backend": "original_torch_mps",
              "request_sha256": sha256_file(args.request),
              "script_sha256": sha256_file(__file__), "listening_status": "pending",
              "comparison": "Independent natural generation; same seed does not imply same RNG trajectory"}
    write_json(args.output / "reference.json", report)
    start = time.perf_counter()
    try:
        with GPUExecution(backend="mps", memory_budget_gib=16, require_ac=True,
                          resource_path=args.output / "resources.jsonl") as guard:
            def cancelled():
                guard.check()
                return False

            with YuE2Pipeline(args.model, args.vae, device="mps", backend="torch-eager",
                              memory_budget_gib=16, vae_core_frames=256) as pipe:
                song = pipe(**request, cancelled=cancelled)
                guard.check()
                song.save_artifacts(args.output / "song")
                report.update(audio_seconds=len(song.audio) / song.sample_rate,
                              truncated=song.truncated, timing=song.timing)
                if any(song.truncated.values()):
                    raise AssertionError("Reference song reached its generation limit")
        report["status"] = "pass_execution_listening_pending"
    except BaseException as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write_json(args.output / "reference.json", report)


if __name__ == "__main__":
    main()
