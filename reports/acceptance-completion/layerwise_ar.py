"""Original full-causal AR arithmetic with one checkpoint layer resident at a time.

This is a separately identified full-prefix reference, not a claim that the
failed original incremental-cache runner fits the budget. MLX is still tested
with its normal cached prefill and final four incremental tokens.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from oracle import _ar_inputs
from lyra.measure import GPUExecution
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.nar import attention
from yue2.storage import model_identity, sha256_file, write_json
from yue2.tokenization_yue2 import YuE2TextTokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--length", type=int, default=12000)
    parser.add_argument("--attention-device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.length <= 12000:
        raise ValueError("Use a new output and length in 5..12000")
    assert version("torch") == "2.11.0" and version("transformers") == "4.57.6"
    args.output.mkdir(parents=True)
    weights = model_identity(args.model)
    ids, input_sequence = _ar_inputs(SimpleNamespace(lengths=[args.length], source=args.source,
        branch="positive", corpus=str(ROOT / "validation/corpus.json"), case="city_lights", mode=None, abc=None),
        YuE2TextTokenizer(args.model / "qwen.tiktoken"))
    length = len(ids)
    positions = np.unique(np.concatenate((np.arange(min(16, length)),
        np.arange(max(0, length // 2 - 4), min(length, length // 2 + 4)),
        np.arange(max(0, length - 16), length))))
    arrays = {f"ids_{length}": np.array(ids, dtype=np.int32), f"positions_{length}": positions}
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    invocation = {"weights": weights, "precision": args.precision, "script_sha256": sha256_file(__file__),
        "input_sequence": input_sequence,
        "input_ids_sha256": hashlib.sha256(arrays[f"ids_{length}"].tobytes()).hexdigest(),
        "method": "one-layer residency; original full-prefix causal attention, query chunks 256; no runtime KV-cache mutation",
        "attention_device": args.attention_device,
        "status": "running", "length": length}
    write_json(args.output / "invocation.json", invocation)
    with GPUExecution(backend="mps", memory_budget_gib=16, require_ac=True,
                      resource_path=args.output / "resources.jsonl") as guard:
        def clear():
            torch.mps.synchronize()
            torch.mps.empty_cache()
            guard.check()

        with torch.device("meta"):
            model = YuE2ForCausalLM(YuE2Config(**json.loads((args.model / "config.json").read_text())))
        model.eval().requires_grad_(False)
        with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            def load(module, prefix):
                state = {name: checkpoint.get_tensor(prefix + name).to(device="mps", dtype=dtype)
                         for name in module.state_dict()}
                module.load_state_dict(state, strict=True, assign=True)
                del state
                clear()

            load(model.model.embed_tokens, "model.embed_tokens.")
            x = model.model.embed_tokens(torch.tensor([ids], device="mps"))
            model.model.embed_tokens.to_empty(device="meta")
            cos, sin = model.model.rotary_emb(torch.arange(length, device="mps")[None])
            selected = torch.tensor(positions, device="mps")
            for index, layer in enumerate(model.model.layers):
                names = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
                for name in names:
                    load(getattr(layer, name), f"model.layers.{index}.{name}.")
                q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
                arrays[f"k_{length}_{index}"] = k.transpose(1, 2).index_select(2, selected).float().cpu().numpy()
                arrays[f"v_{length}_{index}"] = v.transpose(1, 2).index_select(2, selected).float().cpu().numpy()
                attention_inputs = [item[0].to(args.attention_device) for item in (q, k, v)]
                h = attention(*attention_inputs, causal=True, query_chunk_size=256).to("mps")
                del attention_inputs
                x = x + layer.self_attn.o_proj(h.flatten(1)[None])
                del q, k, v, h
                clear()
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                for name in names:
                    getattr(layer, name).to_empty(device="meta")
                clear()
                print("Original layer", index + 1, "/", len(model.model.layers), flush=True)
            load(model.model.norm, "model.norm.")
            # Causal full-prefix positions match the five cached-run outputs.
            hidden = model.model.norm(x[:, -5:])
            del x
            load(model.lm_head, "lm_head.")
            arrays[f"logits_{length}"] = model.lm_head(hidden)[0].float().cpu().numpy()
            clear()
        np.savez(args.output / "ar.npz", **arrays)
        write_json(args.output / "ar.json", {"model": weights, "dtype": str(dtype),
            "input_sequence": input_sequence, "method": invocation["method"], "cases": [{"length": length}]})
        invocation.update(status="complete", arrays_sha256=sha256_file(args.output / "ar.npz"))
        write_json(args.output / "invocation.json", invocation)


if __name__ == "__main__":
    main()
