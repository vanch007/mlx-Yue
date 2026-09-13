"""Original BF16 acoustic modules with phase-specific checkpoint residency."""
import json
from pathlib import Path

from safetensors import safe_open
import torch

from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.nar import CachedNAR, song_chunks

AR = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
NAR = ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp")


@torch.inference_mode()
def synthesize(directory, semantic, config, guard):
    directory = Path(directory)
    with torch.device("meta"):
        model = YuE2ForCausalLM(YuE2Config(**json.loads((directory / "config.json").read_text())))
    model.eval().requires_grad_(False)
    outputs = []

    def clear():
        torch.mps.synchronize()
        torch.mps.empty_cache()
        guard.check()

    def cancelled():
        clear()
        return False

    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        def load(module, prefix):
            guard.check()
            state = {name: checkpoint.get_tensor(prefix + name).to(device="mps", dtype=torch.bfloat16)
                     for name in module.state_dict()}
            module.load_state_dict(state, strict=True, assign=True)
            del state
            clear()

        load(model.model.norm, "model.norm.")
        for name in ("vae2llm", "llm2vae", "time_embedder", "latent_pos_embed"):
            load(getattr(model, name), name + ".")
        for index, chunk in enumerate(song_chunks(semantic.plan.prefix, semantic.tokens,
                                                 semantic.plan.request.seed, config.context)):
            print("Original chunk", index, "prefill", len(chunk.ar_tokens), "frames", len(chunk.noise), flush=True)
            load(model.model.embed_tokens, "model.embed_tokens.")
            for number, layer in enumerate(model.model.layers):
                for name in AR:
                    load(getattr(layer, name), f"model.layers.{number}.{name}.")
            engine = CachedNAR(model, chunk)
            clear()
            model.model.embed_tokens.to_empty(device="meta")
            for layer in model.model.layers:
                for name in AR:
                    getattr(layer, name).to_empty(device="meta")
            clear()
            for number, layer in enumerate(model.model.layers):
                for name in NAR:
                    load(getattr(layer, name), f"model.layers.{number}.{name}.")
            try:
                outputs.append(engine.solve(config.ode_steps, cancelled=cancelled,
                    on_progress=lambda done, total: print("Original midpoint", done, "/", total, flush=True)))
            finally:
                engine.close()
            del engine
            for layer in model.model.layers:
                for name in NAR:
                    getattr(layer, name).to_empty(device="meta")
            clear()
    result = torch.cat(outputs).numpy()
    del model
    clear()
    return result
