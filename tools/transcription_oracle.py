"""Capture identical full-window SheetSage2 inputs with isolated Torch/MLX runtimes.

Torch mode imports the locally reviewed pinned upstream source and loads its
original adapter and encoder weights strictly; it does not execute Hub code.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("torch", "mlx"))
    parser.add_argument("--source", type=Path, default=Path("reports/upstream-sheetsage2"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPZ with waveform and token ids")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.input) as data:
        waveform, ids = data["waveform"], data["ids"]
    start = time.perf_counter()
    if args.backend == "torch":
        import torch
        from safetensors.torch import load_file
        from lyra.measure import GPUExecution
        sys.path.insert(0, str(args.source.resolve()))
        source = importlib.import_module("SheetSage2.modeling_sheetsage2")
        cfg = source.SheetSage2Config(**json.loads((args.model / "config.json").read_text()))
        model = source.SheetSage2Model(cfg)
        weights = load_file(args.model / "model.safetensors")
        for key in ("decoder.embed_tokens.weight", "output_projection.weight"):
            weights[key] = weights["token_embedding.weight"]
        model.load_state_dict(weights, strict=True)
        ec = source.MERT2Config(**cfg.backbone_config)
        ec._attn_implementation = "sdpa"
        model.encoder = source.MERT2Model(ec)
        model.encoder.load_state_dict(load_file(args.parent / "model.safetensors"), strict=True)
        model.merge_lora().eval().requires_grad_(False)
        with GPUExecution(backend="mps", memory_budget_gib=24, require_ac=True), torch.inference_mode():
            model.to("mps")
            memory = model.encode(torch.from_numpy(waveform[None]).to("mps"))
            tokens = torch.from_numpy(ids[None]).to("mps")
            logits, _ = model.decode(memory, tokens)
            _, cache = model.decode(memory, tokens[:, :-1], use_cache=True)
            cached, _ = model.decode(memory, tokens[:, -1:], use_cache=True, past_key_values=cache)
            values = {"memory": memory.cpu().numpy(), "logits": logits.cpu().numpy(),
                      "cached": cached.cpu().numpy()}
    else:
        import mlx.core as mx
        from lyra.measure import GPUExecution
        from lyra.transcription.model import SheetSage2
        with GPUExecution(memory_budget_gib=24, require_ac=True):
            model = SheetSage2.from_pretrained(args.model, args.parent, offline=True)
            memory = model.encode(waveform)
            logits, _ = model.decode(memory, ids.tolist())
            _, cache = model.decode(memory, ids[:-1].tolist())
            cached, _ = model.decode(memory, ids[-1:].tolist(), cache)
            mx.eval(memory, logits, cached)
            values = {"memory": np.array(memory), "logits": np.array(logits), "cached": np.array(cached)}
    np.savez(args.output / "values.npz", **values)
    report = {"backend": args.backend, "elapsed_seconds": time.perf_counter() - start,
              "shapes": {k: list(v.shape) for k, v in values.items()}, "dtype": "float32",
              "encoder_context_seconds": 300, "input_audio_seconds": len(waveform) / 24000,
              "status": "captured_not_accepted"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
