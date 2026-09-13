"""Guarded complete-song reference using the pinned original Torch pipeline."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from lyra.measure import GPUExecution
from yue2.pipeline import YuE2Pipeline
from yue2.storage import sha256_file, write_json


class CapturedPipeline(YuE2Pipeline):
    """Retain reference stages and reclaim unused MPS SDPA workspaces."""

    capture_output: Path

    def generate_semantic(self, plan, **kwargs):
        semantic = super().generate_semantic(plan, **kwargs)
        stage = self.capture_output / "semantic-stage"
        plan.save(stage)
        np.save(stage / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
        write_json(stage / "semantic.json", {
            "timing": semantic.timing, "truncated": semantic.truncated,
            "tokens_sha256": sha256_file(stage / "semantic.npy"),
        })
        return semantic

    def synthesize(self, semantic, **kwargs):
        original = torch.nn.functional.scaled_dot_product_attention

        def bounded_sdpa(*args, **call_kwargs):
            # Keep the original SDPA operands, tiling and arithmetic. These
            # synchronous reference captures are not performance benchmarks.
            torch.mps.synchronize()
            torch.mps.empty_cache()
            result = original(*args, **call_kwargs)
            torch.mps.synchronize()
            torch.mps.empty_cache()
            return result

        torch.nn.functional.scaled_dot_product_attention = bounded_sdpa
        try:
            return super().synthesize(semantic, **kwargs)
        finally:
            torch.nn.functional.scaled_dot_product_attention = original
            torch.mps.synchronize()
            torch.mps.empty_cache()


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
    checkpoints = 0
    try:
        with GPUExecution(backend="mps", memory_budget_gib=16, require_ac=True,
                          resource_path=args.output / "resources.jsonl") as guard:
            def cancelled():
                nonlocal checkpoints
                checkpoints += 1
                # Bound unused MPS allocator buffers during long reference AR
                # runs; this capture is not used as a speed benchmark.
                if checkpoints % 16 == 0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                guard.check()
                return False

            with CapturedPipeline(args.model, args.vae, device="mps", backend="torch-eager",
                                  memory_budget_gib=16, vae_core_frames=256) as pipe:
                pipe.capture_output = args.output
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
