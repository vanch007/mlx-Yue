"""Capture FP32 numerical anchors without a resident FP32 combined generator.

Run in the locked .oracle environment. AR loads only AR parameters. Acoustic
calibration computes FP32 conditioning, releases AR weights, then loads FP32 NAR
weights. The original upstream modules perform every learned operation. This is
accuracy calibration with explicit weight transfers, not a speed benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from lyra.measure import GPUExecution
from yue2.modeling_yue2 import StaticKVCache, YuE2Config, YuE2ForCausalLM
from yue2.nar import CachedNAR, Chunk
from yue2.protocol import CODEC_OFFSET, MUSIC_END
from yue2.storage import model_identity, sha256_file, write_json


AR_PARTS = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
NAR_PARTS = ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp")


def skeleton(directory):
    config = YuE2Config(**json.loads((directory / "config.json").read_text()))
    with torch.device("meta"):
        model = YuE2ForCausalLM(config)
    return model.eval().requires_grad_(False)


def load_module(module, prefix, checkpoint, guard):
    guard.check()
    # Load each module directly from CPU checkpoint storage to FP32 MPS. Never
    # first allocate a full BF16 GPU model and then retain its cast buffers.
    state = {
        name: checkpoint.get_tensor(prefix + name).to(device="mps", dtype=torch.float32)
        for name in module.state_dict()
    }
    module.load_state_dict(state, strict=True, assign=True)
    torch.mps.synchronize()
    del state
    torch.mps.empty_cache()
    guard.check()


def load_ar(model, checkpoint, guard, *, output_head):
    load_module(model.model.embed_tokens, "model.embed_tokens.", checkpoint, guard)
    load_module(model.model.norm, "model.norm.", checkpoint, guard)
    for index, layer in enumerate(model.model.layers):
        for name in AR_PARTS:
            load_module(getattr(layer, name), f"model.layers.{index}.{name}.", checkpoint, guard)
    if output_head:
        load_module(model.lm_head, "lm_head.", checkpoint, guard)


def release_ar(model):
    torch.mps.synchronize()
    model.model.embed_tokens.to_empty(device="meta")
    model.lm_head.to_empty(device="meta")
    for layer in model.model.layers:
        for name in AR_PARTS:
            getattr(layer, name).to_empty(device="meta")
    # Final RMSNorm is also used by NAR and must remain resident.
    torch.mps.empty_cache()


def checkpoint_arrays(output, stage, arrays, *, complete=False):
    path = output / f"{stage}.partial.npz"
    np.savez(path, **arrays)
    if complete:
        path.replace(output / f"{stage}.npz")


@torch.inference_mode()
def capture_ar(args, guard, model, checkpoint):
    load_ar(model, checkpoint, guard, output_head=True)
    arrays = {}
    cases = []
    with np.load(args.oracle / "ar.npz", allow_pickle=False) as reference, ExitStack() as workspace_hooks:
        def clear_workspace(_module, _inputs, _output):
            torch.mps.synchronize()
            torch.mps.empty_cache()

        for layer in model.model.layers:
            workspace_hooks.callback(layer.register_forward_hook(clear_workspace).remove)
        available = sorted(int(key.removeprefix("ids_")) for key in reference.files if key.startswith("ids_"))
        lengths = available if args.lengths is None else args.lengths
        if any(length not in available for length in lengths):
            raise ValueError("Calibration lengths must exist in the saved BF16 reference")
        for length in lengths:
            guard.check()
            ids = reference[f"ids_{length}"]
            cache = StaticKVCache(model.config.num_hidden_layers, 1, model.config.num_key_value_heads,
                                  length, model.config.head_dim, torch.float32, torch.device("mps"))
            try:
                tokens = torch.tensor(ids[None], dtype=torch.long, device="mps")
                for start in range(0, length - 4, args.prefill_chunk_size):
                    guard.check()
                    stop = min(start + args.prefill_chunk_size, length - 4)
                    result = model(tokens[:, start:stop], past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    torch.mps.synchronize()
                    if stop < length - 4:
                        del result
                        torch.mps.empty_cache()
                logits = [result.logits[0, -1].cpu().numpy()]
                del result
                for index in range(length - 4, length):
                    guard.check()
                    result = model(tokens[:, index:index + 1], past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    logits.append(result.logits[0, -1].cpu().numpy())
                    del result
                arrays[f"ids_{length}"] = ids
                arrays[f"logits_{length}"] = np.stack(logits)
                positions = reference[f"positions_{length}"]
                arrays[f"positions_{length}"] = positions
                selected = torch.tensor(positions, dtype=torch.long, device="mps")
                for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
                    arrays[f"k_{length}_{layer}"] = key.index_select(2, selected).cpu().numpy()
                    arrays[f"v_{length}_{layer}"] = value.index_select(2, selected).cpu().numpy()
                cases.append(length)
                checkpoint_arrays(args.output, "ar", arrays)
                write_json(args.output / "progress.json", {"completed_lengths": cases, "complete": False})
                print(json.dumps({"completed_length": length}), flush=True)
            finally:
                cache.key_cache.clear()
                cache.value_cache.clear()
                torch.mps.synchronize()
                torch.mps.empty_cache()
    checkpoint_arrays(args.output, "ar", arrays, complete=True)
    return {"lengths": cases, "prefill_chunk_size": args.prefill_chunk_size,
            "storage_policy": "unused MPS workspaces cleared after each AR layer"}


def raw_time(t):
    return torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()


@torch.inference_mode()
def capture_nar(args, guard, model, checkpoint):
    with np.load(args.oracle / "nar.npz", allow_pickle=False) as reference:
        codec = reference["codec"].tolist()
        if not 1 <= len(codec) <= 500:
            raise ValueError("Acoustic calibration requires a short 1..500 frame reference")
        prefix = reference["prefix"].tolist()
        noise = reference["noise"].copy()
        for name in ("vae2llm", "llm2vae", "time_embedder", "latent_pos_embed"):
            load_module(getattr(model, name), name + ".", checkpoint, guard)
        load_ar(model, checkpoint, guard, output_head=False)
        chunk = Chunk(prefix + [token + CODEC_OFFSET for token in codec] + [MUSIC_END],
                      torch.from_numpy(noise))
        engine = CachedNAR(model, chunk, query_chunk_size=256)
        arrays = {"prefix": np.asarray(prefix, dtype=np.int32), "codec": np.asarray(codec, dtype=np.int32),
                  "noise": noise}
        try:
            for layer, (key, value) in enumerate(engine.cache):
                arrays[f"k_{layer}"] = key.permute(1, 0, 2)[None].cpu().numpy()
                arrays[f"v_{layer}"] = value.permute(1, 0, 2)[None].cpu().numpy()
            release_ar(model)
            guard.check()
            for index, layer in enumerate(model.model.layers):
                for name in NAR_PARTS:
                    load_module(getattr(layer, name), f"model.layers.{index}.{name}.", checkpoint, guard)
            state = torch.from_numpy(noise).to("mps")
            for step in range(32):
                guard.check()
                first = engine.velocity(state, raw_time(1 - step / 32))
                mid = state - first / 64
                velocity = engine.velocity(mid, raw_time(1 - step / 32 - 1 / 64))
                if step in (0, 1, 15, 31):
                    for name, value in (("state", state), ("velocity", first),
                                        ("mid", mid), ("mid_velocity", velocity)):
                        arrays[f"{name}_{step}"] = value.cpu().numpy()
                    checkpoint_arrays(args.output, "nar", arrays)
                state = state - velocity / 32
                torch.mps.synchronize()
            arrays["latents"] = state.cpu().numpy()
            # Fixed BF16 states separate operator error from accumulated ODE error.
            for step in (0, 1, 15, 31):
                for name, state_name, t in (
                    (f"fixed_velocity_{step}", f"state_{step}", 1 - step / 32),
                    (f"fixed_mid_velocity_{step}", f"mid_{step}", 1 - step / 32 - 1 / 64),
                ):
                    guard.check()
                    fixed = torch.from_numpy(reference[state_name].copy()).to("mps")
                    arrays[name] = engine.velocity(fixed, raw_time(t)).cpu().numpy()
            checkpoint_arrays(args.output, "nar", arrays, complete=True)
        finally:
            engine.close()
    return {"frames": len(codec), "steps": 32, "ar_weights_released_before_velocity": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ar", "nar"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--memory-budget-gib", type=float, default=16)
    parser.add_argument("--require-ac", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.prefill_chunk_size <= 1024:
        parser.error("FP32 calibration prefill chunks must be in 1..1024")
    if version("torch") != "2.11.0" or version("transformers") != "4.57.6":
        raise RuntimeError("Use the pinned .oracle environment")
    import os
    for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        if os.environ.get(name) == "1":
            raise RuntimeError(f"Disable {name} for numerical calibration")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Calibration requires a new evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "invocation.json", {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "script_sha256": sha256_file(__file__), "precision": "float32",
        "upstream_commit": "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e",
        "runtime": {name: version(name) for name in ("torch", "transformers", "numpy")},
        "torch_git_version": torch.version.git_version,
        "reference_sha256": sha256_file(args.oracle / f"{args.stage}.npz"),
        "weights": model_identity(args.model),
        "purpose": "independent_precision_calibration_not_performance_benchmark",
    })
    torch.set_float32_matmul_precision("highest")
    with GPUExecution(backend="mps", memory_budget_gib=args.memory_budget_gib,
                      resource_path=args.output / "resources.jsonl", require_ac=args.require_ac) as guard:
        model = skeleton(args.model)
        try:
            with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as checkpoint:
                result = {"ar": capture_ar, "nar": capture_nar}[args.stage](args, guard, model, checkpoint)
            guard.check()
        finally:
            model.to_empty(device="meta")
            del model
            torch.mps.synchronize()
            torch.mps.empty_cache()
    write_json(args.output / "progress.json", {"complete": True, **result})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
