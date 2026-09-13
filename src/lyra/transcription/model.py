"""Native MLX SheetSage2, including its MERT2 LoRA encoder and BART decoder."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.numpy import load_file

from ..bart import BartDecoder
from ..mert import MERT2, convert_mert_weights
from yue2.storage import sha256_file
from .control import check_cancelled

MODEL_REPO = "m-a-p/SheetSage2"
MODEL_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"
MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"


def resolve_models(model=MODEL_REPO, base_model=MERT_REPO, *, offline=False, cache_dir=None):
    from huggingface_hub import snapshot_download
    result = []
    for source, repo, revision in ((model, MODEL_REPO, MODEL_REVISION), (base_model, MERT_REPO, MERT_REVISION)):
        if Path(source).is_dir():
            result.append(Path(source))
        else:
            if str(source) != repo:
                raise ValueError(f"Use the pinned {repo} or a reviewed local directory")
            result.append(Path(snapshot_download(repo, revision=revision, local_files_only=offline,
                               cache_dir=cache_dir, allow_patterns=["config.json", "model.safetensors"])))
    return tuple(result)


class SheetSage2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MERT2(config["backbone_config"])
        self.layer_weight = mx.zeros((config["backbone_config"]["num_hidden_layers"] + 1,))
        self.encoder_projection = nn.Linear(config["backbone_config"]["hidden_size"], config["hidden_size"])
        self.token_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.decoder = BartDecoder(config)
        object.__setattr__(self, "config", config)

    @classmethod
    def from_pretrained(cls, model=MODEL_REPO, base_model=MERT_REPO, *, cancelled=None, **kwargs):
        check_cancelled(cancelled)
        directory, parent = resolve_models(model, base_model, **kwargs)
        check_cancelled(cancelled)
        config = json.loads((directory / "config.json").read_text())
        if config["weights_format"] != "adapter":
            raise ValueError("Expected the pinned SheetSage2 adapter checkpoint")
        parent_file = parent / "model.safetensors"
        parent_hash = sha256_file(parent_file)
        check_cancelled(cancelled)
        if parent_hash != config["base_model_sha256"]:
            raise ValueError("MERT2 checkpoint does not match SheetSage2's pinned parent")
        parent_config = json.loads((parent / "config.json").read_text())
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                    "sampling_rate", "hop_length", "n_fft", "win_length", "num_mel_bins",
                    "conv_depthwise_kernel_size", "rotary_embedding_base", "subsampling_channels",
                    "subsampling_depths", "layer_norm_eps", "subsampling_layer_norm_eps", "variant"):
            if config["backbone_config"][key] != parent_config[key]:
                raise ValueError(f"MERT2 architecture mismatch: {key}")
        adapter, weights = load_file(directory / "model.safetensors"), load_file(parent_file)
        check_cancelled(cancelled)
        scale = config["lora_alpha"] / config["lora_rank"]
        # The official loader merges LoRA on CPU in FP32 before inference.
        for key in list(adapter):
            check_cancelled(cancelled)
            if key.startswith("adapter.") and key.endswith("lora_A.weight"):
                target = key.removeprefix("adapter.").removesuffix("lora_A.weight") + "weight"
                bkey = key.replace("lora_A.weight", "lora_B.weight")
                weights[target] += (adapter[bkey] @ adapter[key]) * scale
        instance = cls(config)
        instance.encoder.load_weights(convert_mert_weights(weights), strict=True)
        own = [(key, mx.array(value)) for key, value in adapter.items() if not key.startswith("adapter.")]
        # Load the decoder/projection subset strictly, with the already loaded encoder retained.
        from mlx.utils import tree_flatten
        own += [("encoder." + key, value) for key, value in tree_flatten(instance.encoder.parameters())]
        instance.load_weights(own, strict=True)
        mx.eval(instance.parameters())
        check_cancelled(cancelled)
        object.__setattr__(instance, "identity", {
            "model": MODEL_REPO, "revision": MODEL_REVISION,
            "model_sha256": sha256_file(directory / "model.safetensors"),
            "config_sha256": sha256_file(directory / "config.json"),
            "base_model": MERT_REPO, "base_revision": MERT_REVISION, "base_sha256": parent_hash,
            "backend": "mlx", "dtype": "float32", "lora_merge": "numpy_cpu_fp32",
            "implementation_sha256": {str(p.relative_to(Path(__file__).parent.parent)): sha256_file(p)
                                      for p in [Path(__file__), Path(__file__).with_name("pipeline.py"),
                                                Path(__file__).with_name("control.py"),
                                                Path(__file__).parent.parent / "mert.py",
                                                Path(__file__).parent.parent / "bart.py"]},
        })
        return instance

    def encode(self, waveform, cancelled=None):
        check_cancelled(cancelled)
        window = round(self.config["input_audio_length"] * self.config["sampling_rate"])
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not 1025 <= len(waveform) <= window or not np.isfinite(waveform).all():
            raise ValueError("Expected finite mono audio inside one SheetSage2 window")
        # Preserve official full-window silence context, even for short recordings.
        waveform = np.pad(waveform, (0, window - len(waveform)))
        mixed = self.encoder(mx.array(waveform[None]), mx.softmax(self.layer_weight), cancelled)
        return self.encoder_projection(mixed)

    def decode(self, memory, ids, cache=None):
        hidden, cache = self.decoder(self.token_embedding(mx.array([ids])), memory, cache)
        return self.token_embedding.as_linear(hidden), cache
