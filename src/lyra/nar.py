"""Faithful BF16 MLX acoustic flow matching for YuE2.

The acoustic checkpoint contains only the learned NAR experts and auxiliary
layers.  Causal conditioning and the final RMS norm are borrowed from the
shared, unquantized BF16 Qwen3 AR model; those weights are never duplicated.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
from dataclasses import dataclass
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3

from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, chunk_ranges
from .ar import SourceMLP, SourceRMSNorm, SourceRoPE, _source_sdpa, _source_silu

_LATENT_DIM = 64
_TIME_EMBED_DIM = 256


@dataclass
class Chunk:
    ar_tokens: list[int]
    noise: np.ndarray
    nar_cond_end: int = 0


def _integers(values, name):
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} must be a nonempty sequence of integer token IDs")
    return [int(v) for v in result]


def _logit(t):
    if t <= 0:
        return -20.0
    if t >= 1:
        return 20.0
    return max(-20.0, min(20.0, math.log(t / (1.0 - t))))


class _AcousticAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_heads: tuple[int, int, int],
        eps: float,
    ):
        super().__init__()
        num_heads, num_kv_heads, head_dim = intermediate_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = SourceRMSNorm(mx.ones((head_dim,)), eps)
        self.k_norm = SourceRMSNorm(mx.ones((head_dim,)), eps)


class _AcousticLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        heads = (
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
        )
        eps = config["rms_norm_eps"]
        self.nar_input_layernorm = SourceRMSNorm(mx.ones((hidden_size,)), eps)
        self.nar_self_attn = _AcousticAttention(hidden_size, heads, eps)
        self.nar_pre_mlp_layernorm = SourceRMSNorm(mx.ones((hidden_size,)), eps)
        self.nar_mlp = SourceMLP(qwen3.MLP(hidden_size, config["intermediate_size"]))


class _AcousticBackbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.layers = [_AcousticLayer(config) for _ in range(config["num_hidden_layers"])]


class _TimestepEmbedder(nn.Module):
    """FP32 sinusoid followed by the checkpoint's BF16 two-layer MLP."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = [
            nn.Linear(_TIME_EMBED_DIM, hidden_size),
            _source_silu,
            nn.Linear(hidden_size, hidden_size),
        ]
        half = _TIME_EMBED_DIM // 2
        frequencies = mx.exp(
            mx.array(-math.log(10000.0), dtype=mx.float32)
            * mx.arange(half, dtype=mx.float32)
            / half
        )
        mx.eval(frequencies)
        # This is a derived constant, not a checkpoint parameter.
        object.__setattr__(self, "_frequencies", frequencies)

    def __call__(self, timestep: mx.array) -> mx.array:
        arguments = timestep.astype(mx.float32)[..., None] * self._frequencies[None, :]
        embedding = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
        embedding = embedding.astype(mx.bfloat16)
        hidden = self.mlp[0](embedding)
        hidden = self.mlp[1](hidden)
        return self.mlp[2](hidden)


class _AudioPositionEmbedding(nn.Module):
    """The precomputed positional buffer is loaded verbatim from the checkpoint."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden_size), dtype=mx.bfloat16)

    def __call__(self, position_ids: mx.array) -> mx.array:
        return self.pe[position_ids]


class AcousticModel(nn.Module):
    """Acoustic-only learned modules plus a non-owning reference to BF16 AR."""

    def __init__(self, config: dict, ar_model):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.model = _AcousticBackbone(config)
        self.llm2vae = nn.Linear(hidden_size, config["latent_dim"])
        self.vae2llm = nn.Linear(config["latent_dim"], hidden_size)
        self.time_embedder = _TimestepEmbedder(hidden_size)
        self.latent_pos_embed = _AudioPositionEmbedding(
            config["max_latent_frames"], hidden_size
        )

        inverse_frequency = SourceRoPE(
            config["head_dim"], config["rope_theta"]
        ).inv_freq
        shift = mx.array(config["timestep_shift"], dtype=mx.float32)
        shift_delta = mx.array(config["timestep_shift"] - 1, dtype=mx.float32)
        one = mx.array(1.0, dtype=mx.bfloat16)
        mx.eval(inverse_frequency, shift, shift_delta, one)

        # nn.Module registers arrays and nested dictionaries as state.  These
        # values are runtime metadata/derived constants, while the AR reference
        # must not make its weights part of this module's parameter tree.
        object.__setattr__(self, "config", MappingProxyType(dict(config)))
        object.__setattr__(self, "_ar_model", ar_model)
        object.__setattr__(self, "_rope_inverse_frequency", inverse_frequency)
        object.__setattr__(self, "_timestep_shift", shift)
        object.__setattr__(self, "_timestep_shift_delta", shift_delta)
        object.__setattr__(self, "_one", one)
        object.__setattr__(self, "manifest", None)

    @property
    def ar_model(self):
        return self._ar_model

    def rope_factors(self, start: int, length: int) -> tuple[mx.array, mx.array]:
        """Compute global-position trigonometry in FP32, then round to BF16."""
        positions = mx.arange(start, start + length, dtype=mx.float32)
        angles = positions[:, None] * self._rope_inverse_frequency[None, :]
        cos = mx.cos(angles).astype(mx.bfloat16)[None, None, :, :]
        sin = mx.sin(angles).astype(mx.bfloat16)[None, None, :, :]
        return cos, sin

    def shift_timestep(self, raw_t: float) -> mx.array:
        # Quantize the input first, but use source FP32 opmath for sigmoid and
        # wrapped Python coefficients. Each tensor result still rounds to BF16.
        raw = mx.array(raw_t, dtype=mx.bfloat16).astype(mx.float32)
        t_sigmoid = mx.sigmoid(raw).astype(mx.bfloat16).astype(mx.float32)
        numerator = (self._timestep_shift * t_sigmoid).astype(mx.bfloat16)
        delta = (self._timestep_shift_delta * t_sigmoid).astype(mx.bfloat16)
        return numerator / (self._one + delta)


def _config_int(config: dict, name: str, *, minimum: int = 1) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"config.json field {name!r} must be an integer >= {minimum}")
    return int(value)


def _config_float(config: dict, name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"config.json field {name!r} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"config.json field {name!r} must be a finite positive number")
    return value


def _read_config(directory: Path) -> dict:
    path = directory / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing generator configuration: {path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid generator configuration: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")

    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "latent_dim",
        "max_latent_frames",
    ):
        config[name] = _config_int(config, name)
    for name in ("rms_norm_eps", "rope_theta", "timestep_shift"):
        config[name] = _config_float(config, name)

    if config.get("latent_type") != "vae":
        raise ValueError("YuE2 acoustic loading requires config latent_type='vae'")
    if config["latent_dim"] != _LATENT_DIM:
        raise ValueError(f"YuE2 acoustic loading requires latent_dim={_LATENT_DIM}")
    if config["head_dim"] % 2:
        raise ValueError("YuE2 RoPE requires an even head_dim")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if config.get("tie_word_embeddings") is not False:
        raise ValueError("YuE2 requires untied token embeddings")
    return config


def _shared_ar_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    q_width = config["num_attention_heads"] * config["head_dim"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config["vocab_size"], hidden),
        "model.norm.weight": (hidden,),
    }
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "self_attn.q_proj.weight": (q_width, hidden),
                prefix + "self_attn.k_proj.weight": (kv_width, hidden),
                prefix + "self_attn.v_proj.weight": (kv_width, hidden),
                prefix + "self_attn.o_proj.weight": (hidden, q_width),
                prefix + "self_attn.q_norm.weight": (config["head_dim"],),
                prefix + "self_attn.k_norm.weight": (config["head_dim"],),
                prefix + "post_attention_layernorm.weight": (hidden,),
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return shapes


def _validate_shared_ar(ar_model, config: dict) -> None:
    args = getattr(ar_model, "args", None)
    if args is None:
        raise TypeError("ar_model must be an mlx_lm Qwen3-compatible model")
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "tie_word_embeddings",
    ):
        if not hasattr(args, name) or getattr(args, name) != config[name]:
            raise ValueError(f"Shared AR configuration does not match generator field {name!r}")
    if getattr(args, "rope_scaling", None) not in (None, {}):
        raise ValueError("YuE2 acoustic conditioning does not support scaled RoPE")
    backbone = getattr(ar_model, "model", None)
    if (
        backbone is None
        or len(getattr(backbone, "layers", ())) != config["num_hidden_layers"]
    ):
        raise ValueError("Shared AR model has an incompatible transformer layer count")
    if not isinstance(backbone.norm, SourceRMSNorm):
        raise TypeError("Shared AR final norm does not preserve YuE2 source arithmetic")
    for layer in backbone.layers:
        attention = layer.self_attn
        if (
            not isinstance(layer.input_layernorm, SourceRMSNorm)
            or not isinstance(layer.post_attention_layernorm, SourceRMSNorm)
            or not isinstance(attention.q_norm, SourceRMSNorm)
            or not isinstance(attention.k_norm, SourceRMSNorm)
            or not isinstance(attention.rope, SourceRoPE)
            or not isinstance(layer.mlp, SourceMLP)
        ):
            raise TypeError("Shared AR layers do not preserve YuE2 source arithmetic")

    try:
        parameters = dict(tree_flatten(ar_model.parameters()))
    except (AttributeError, TypeError) as error:
        raise TypeError("ar_model must expose MLX parameters") from error
    for name, shape in _shared_ar_shapes(config).items():
        value = parameters.get(name)
        if value is None:
            raise ValueError(f"Shared AR model is missing conditioning tensor {name!r}")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"Shared AR tensor {name!r} has shape {tuple(value.shape)}, expected {shape}"
            )
        if value.dtype != mx.bfloat16:
            raise ValueError(
                f"Shared AR tensor {name!r} must be BF16; quantized AR cannot condition NAR"
            )


def _validate_acoustic_weights(model: AcousticModel, weights: dict[str, mx.array]) -> None:
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(expected.keys() - weights.keys())
    extra = sorted(weights.keys() - expected.keys())
    if missing:
        raise ValueError("Missing acoustic tensors: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected acoustic tensors: " + ", ".join(extra))
    for name, destination in expected.items():
        source = weights[name]
        if not isinstance(source, mx.array):
            raise TypeError(f"Acoustic tensor {name!r} is not an MLX array")
        if tuple(source.shape) != tuple(destination.shape):
            raise ValueError(
                f"Acoustic tensor {name!r} has shape {tuple(source.shape)}, "
                f"expected {tuple(destination.shape)}"
            )
        if source.dtype != mx.bfloat16:
            raise ValueError(f"Acoustic tensor {name!r} must be BF16, got {source.dtype}")


def _read_manifest(directory: Path, verify: bool) -> dict:
    if verify:
        from .conversion import verify_conversion

        manifest = verify_conversion(directory)
    else:
        path = directory / "conversion.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing conversion manifest: {path}") from None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid conversion manifest: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("conversion.json must be a schema-1 manifest")
    precisions = manifest.get("precisions")
    precision = precisions.get("bf16") if isinstance(precisions, dict) else None
    expected = {
        "ar": "ar-bf16.safetensors",
        "nar": "nar-bf16.safetensors",
        "dtype": "bfloat16",
    }
    if precision != expected:
        raise ValueError("BF16 conversion manifest entry is incompatible")
    return manifest


def load_nar(directory, ar_model, *, verify: bool = True) -> AcousticModel:
    """Strictly load acoustic BF16 weights and bind a shared BF16 AR model.

    ``verify=True`` validates the complete conversion manifest and file hashes
    before any tensors are accepted.  Disabling hash verification never
    relaxes key, shape, dtype, configuration, or shared-AR validation.
    """
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Converted model directory does not exist: {directory}")

    manifest = _read_manifest(directory, verify)

    config = _read_config(directory)
    _validate_shared_ar(ar_model, config)
    acoustic_path = directory / "nar-bf16.safetensors"
    if not acoustic_path.is_file():
        raise FileNotFoundError(f"Missing acoustic checkpoint: {acoustic_path}")
    try:
        loaded = mx.load(str(acoustic_path))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unable to load acoustic checkpoint: {acoustic_path}") from error
    if not isinstance(loaded, dict):
        raise ValueError("nar-bf16.safetensors must contain named tensors")
    weights = dict(loaded)

    model = AcousticModel(config, ar_model)
    _validate_acoustic_weights(model, weights)
    model.load_weights(list(weights.items()), strict=True)
    del loaded, weights
    model.eval()
    model.freeze()
    ar_model.eval()
    mx.eval(model.parameters())
    object.__setattr__(model, "manifest", manifest)
    return model


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    first_cos = (first * cos).astype(x.dtype)
    second_sin = (second * sin).astype(x.dtype)
    second_cos = (second * cos).astype(x.dtype)
    first_sin = (first * sin).astype(x.dtype)
    rotated_first = (first_cos - second_sin).astype(x.dtype)
    rotated_second = (second_cos + first_sin).astype(x.dtype)
    return mx.concatenate([rotated_first, rotated_second], axis=-1)


def _project_qkv(
    attention,
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[mx.array, mx.array, mx.array]:
    batch, length, _ = x.shape
    query = attention.q_proj(x).reshape(batch, length, num_heads, head_dim)
    key = attention.k_proj(x).reshape(batch, length, num_kv_heads, head_dim)
    value = attention.v_proj(x).reshape(batch, length, num_kv_heads, head_dim)
    query = attention.q_norm(query).transpose(0, 2, 1, 3)
    key = attention.k_norm(key).transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin), value


def _attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    causal: bool,
    query_chunk_size: int | None,
) -> mx.array:
    """Native GQA over complete keys, optionally tiled only along queries."""
    if query.ndim != 4 or key.ndim != 4 or tuple(value.shape) != tuple(key.shape):
        raise ValueError("Expected attention Q/K/V shaped [batch,heads,tokens,dim]")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Attention batch and head dimensions do not match")
    if query.shape[1] % key.shape[1]:
        raise ValueError("Invalid grouped-query head count")
    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("Attention sequences must be nonempty")
    if causal and query.shape[2] != key.shape[2]:
        raise ValueError("Causal prefill requires equal query and key lengths")

    block = query.shape[2] if query_chunk_size is None else query_chunk_size
    # Convert temporary operands once per layer, not once per query tile. The
    # retained conditioning cache and the solver state remain BF16.
    needs_fp32 = causal or min(block, query.shape[2]) > 8
    precise_key = key.astype(mx.float32) if needs_fp32 else key
    precise_value = value.astype(mx.float32) if needs_fp32 else value
    outputs = []
    scale = query.shape[-1] ** -0.5
    for start in range(0, query.shape[2], block):
        end = min(start + block, query.shape[2])
        if causal:
            # MLX's causal mask is lower-right aligned.  Restricting keys to
            # [:end] therefore gives this tile its absolute causal positions.
            used_key = precise_key[..., :end, :]
            used_value = precise_value[..., :end, :]
            mask = "causal"
        else:
            # Every acoustic tile retains the complete conditioning + acoustic
            # key set; this is memory tiling, not local attention.
            vector = end - start <= 8
            used_key = key if vector else precise_key
            used_value = value if vector else precise_value
            mask = None
        outputs.append(
            _source_sdpa(
                query[..., start:end, :],
                used_key,
                used_value,
                scale=scale,
                mask=mask,
            )
        )
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def _cpu_float32_noise(value, name: str) -> np.ndarray:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a CPU float32 tensor")
        if value.requires_grad:
            value = value.detach()
        array = value.numpy()
    elif isinstance(value, np.ndarray):
        if value.dtype != np.float32:
            raise ValueError(f"{name} must have dtype float32")
        array = value
    else:
        raise TypeError(f"{name} must be a NumPy array or CPU PyTorch tensor")
    if array.ndim != 2 or array.shape[1] != _LATENT_DIM or array.shape[0] < 1:
        raise ValueError(f"{name} must be nonempty [frames,{_LATENT_DIM}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class CachedNAR:
    """One original acoustic chunk with invariant causal conditioning.

    ``cache`` is a list of ``(key, value)`` pairs, one per transformer layer.
    Every tensor has layout ``[1, num_key_value_heads, visible_ar_tokens,
    head_dim]``.  This stable layout is intentionally exposed for oracle
    validation of conditioning K/V; acoustic keys are never stored in it.
    """

    def __init__(self, model: AcousticModel, chunk: Chunk, *, query_chunk_size=None, cancelled=None):
        if not isinstance(model, AcousticModel):
            raise TypeError("CachedNAR requires the model returned by load_nar")
        if query_chunk_size is not None:
            query_chunk_size = _positive_integer(query_chunk_size, "query_chunk_size")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        self._cancelled = cancelled
        self.model = model
        self.chunk = chunk
        self.query_chunk_size = query_chunk_size
        self.dtype = mx.bfloat16
        self.cache: list[tuple[mx.array, mx.array]] = []
        self.cos: mx.array | None = None
        self.sin: mx.array | None = None
        self.pos_emb: mx.array | None = None
        self._initial_state: mx.array | None = None
        self._closed = False

        noise = _cpu_float32_noise(chunk.noise, "chunk.noise")
        tokens = list(chunk.ar_tokens)
        if not tokens or any(
            isinstance(token, bool) or not isinstance(token, Integral) for token in tokens
        ):
            raise ValueError("chunk.ar_tokens must be a nonempty integer sequence")
        tokens = [int(token) for token in tokens]
        if min(tokens) < 0 or max(tokens) >= model.config["vocab_size"]:
            raise ValueError("AR conditioning tokens are outside the model vocabulary")
        if isinstance(chunk.nar_cond_end, bool) or not isinstance(chunk.nar_cond_end, Integral):
            raise ValueError("nar_cond_end must be a nonnegative integer")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be a nonnegative integer")

        self.ar_length = len(tokens)
        self.nar_length = len(noise) + 2
        if self.ar_length + self.nar_length > model.config["max_position_embeddings"]:
            raise ValueError("Original acoustic chunk exceeds the model context")
        self.visible_length = (
            min(int(chunk.nar_cond_end), self.ar_length)
            if chunk.nar_cond_end
            else self.ar_length
        )
        object.__setattr__(self, "_tokens", tokens)
        self._initial_state = mx.array(noise, dtype=mx.bfloat16)
        self.cos, self.sin = model.rope_factors(self.ar_length, self.nar_length)
        local_positions = mx.minimum(
            mx.arange(self.nar_length, dtype=mx.int32),
            mx.array(model.config["max_latent_frames"] - 1, dtype=mx.int32),
        )
        self.pos_emb = model.latent_pos_embed(local_positions)[None, :, :]
        mx.eval(self._initial_state, self.cos, self.sin, self.pos_emb)
        try:
            self._prefill()
        except BaseException:
            self.close()
            raise

    def _project(self, attention, x, cos, sin):
        config = self.model.config
        return _project_qkv(
            attention,
            x,
            cos,
            sin,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
        )

    def _attend(self, query, key, value, *, causal=False):
        return _attention(
            query,
            key,
            value,
            causal=causal,
            query_chunk_size=self.query_chunk_size,
        )

    def _prefill(self) -> None:
        ar_backbone = self.model.ar_model.model
        ids = mx.array(np.asarray(self._tokens, dtype=np.int32))[None, :]
        cos, sin = self.model.rope_factors(0, self.ar_length)
        x = ar_backbone.embed_tokens(ids)
        for layer in ar_backbone.layers:
            if self._cancelled is not None and self._cancelled():
                raise InterruptedError("Cancelled during acoustic prefill")
            query, key, value = self._project(
                layer.self_attn, layer.input_layernorm(x), cos, sin
            )
            cached_key = mx.contiguous(key[..., : self.visible_length, :])
            cached_value = mx.contiguous(value[..., : self.visible_length, :])
            attention = self._attend(query, key, value, causal=True)
            attention = attention.transpose(0, 2, 1, 3).reshape(
                1, self.ar_length, -1
            )
            x = x + layer.self_attn.o_proj(attention)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            # Materialize each immutable layer cache and the next hidden state.
            # This keeps the one-time prefill graph from retaining every layer;
            # no value is transferred to CPU.
            mx.eval(cached_key, cached_value, x)
            self.cache.append((cached_key, cached_value))
        mx.eval(cos, sin)

    def _state_array(self, state) -> mx.array:
        if isinstance(state, mx.array):
            if tuple(state.shape) != (self.nar_length - 2, _LATENT_DIM):
                raise ValueError("ODE state shape changed")
            if not mx.issubdtype(state.dtype, mx.floating):
                raise TypeError("ODE state must be floating point")
            return state if state.dtype == mx.bfloat16 else state.astype(mx.bfloat16)
        array = _cpu_float32_noise(state, "state")
        if tuple(array.shape) != (self.nar_length - 2, _LATENT_DIM):
            raise ValueError("ODE state shape changed")
        return mx.array(array, dtype=mx.bfloat16)

    def velocity(self, state, raw_t) -> mx.array:
        """Evaluate only acoustic experts; returns BF16 ``[frames,64]``."""
        if self._closed:
            raise RuntimeError("CachedNAR is closed")
        if isinstance(raw_t, bool) or not isinstance(raw_t, Real) or not math.isfinite(raw_t):
            raise ValueError("raw_t must be a finite real number")
        state = self._state_array(state)
        boundary_state = mx.pad(state, ((1, 1), (0, 0)))
        shifted = self.model.shift_timestep(float(raw_t))
        x = self.model.vae2llm(boundary_state[None, :, :])

        # Upstream expands the identical shifted timestep to every acoustic
        # position before the learned BF16 MLP.  Keep that batched arithmetic;
        # only its FP32 frequency vector is precomputed.
        timesteps = mx.broadcast_to(shifted, (self.nar_length,))
        time_embedding = self.model.time_embedder(timesteps)[None, :, :]
        x = x + time_embedding
        x = x + self.pos_emb

        for nar_layer, (ar_key, ar_value) in zip(self.model.model.layers, self.cache):
            if self._cancelled is not None and self._cancelled():
                raise InterruptedError("Cancelled during acoustic velocity")
            query, key, value = self._project(
                nar_layer.nar_self_attn,
                nar_layer.nar_input_layernorm(x),
                self.cos,
                self.sin,
            )
            key = mx.concatenate([ar_key, key], axis=2)
            value = mx.concatenate([ar_value, value], axis=2)
            attention = self._attend(query, key, value)
            attention = attention.transpose(0, 2, 1, 3).reshape(
                1, self.nar_length, -1
            )
            x = x + nar_layer.nar_self_attn.o_proj(attention)
            x = x + nar_layer.nar_mlp(nar_layer.nar_pre_mlp_layernorm(x))

        x = self.model.ar_model.model.norm(x)
        return self.model.llm2vae(x)[0, 1:-1, :]

    def solve(
        self,
        steps=32,
        cancelled: Callable[[], bool] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Run the reference midpoint solver (two velocities per step)."""
        if self._closed:
            raise RuntimeError("CachedNAR is closed")
        steps = _positive_integer(steps, "steps")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if on_progress is not None and not callable(on_progress):
            raise TypeError("on_progress must be callable")

        state = self._initial_state
        dt = 1.0 / steps
        # Python scalar operands use FP32 opmath in the source, then round each
        # product to BF16. Rounding dt itself breaks non-power-of-two step counts.
        half_dt = mx.array(dt / 2.0, dtype=mx.float32)
        full_dt = mx.array(dt, dtype=mx.float32)
        mx.eval(half_dt, full_dt)
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = _logit(t)
            first = self.velocity(state, raw)
            mx.eval(first)
            midpoint = state - (first.astype(mx.float32) * half_dt).astype(mx.bfloat16)

            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            mid_t = t - dt / 2.0
            raw_mid = _logit(mid_t)
            velocity = self.velocity(midpoint, raw_mid)
            state = state - (velocity.astype(mx.float32) * full_dt).astype(mx.bfloat16)
            # Materializing after every submitted midpoint step prevents the
            # lazy graph from spanning subsequent velocity evaluations.
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, steps)

        result = np.array(state.astype(mx.float32), dtype=np.float32, copy=True)
        if result.shape != (self.nar_length - 2, _LATENT_DIM):
            raise RuntimeError("Acoustic solver returned an invalid latent shape")
        if not np.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self) -> None:
        self.cache.clear()
        self.cos = None
        self.sin = None
        self.pos_emb = None
        self._initial_state = None
        self._closed = True


def _chunks_with_supplied_noise(prefix, codec, seed, context, noise) -> list[Chunk]:
    """Use the native validation/cuts without allocating an unused random song."""
    prefix, codec = _integers(prefix, "prefix"), _integers(codec, "codec")
    if min(prefix) < 0 or min(codec) < 0 or max(codec) >= CODEC_SIZE:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(context, bool) or not isinstance(context, Integral) or not 1 <= context <= CONTEXT:
        raise ValueError("context must be an integer in 1..24576")
    ranges = chunk_ranges(len(codec), len(prefix), int(context))
    full_noise = _cpu_float32_noise(noise, "noise")
    if len(full_noise) != len(codec):
        raise ValueError("Supplied noise must have one frame per semantic token")
    return [
        Chunk(prefix + [value + CODEC_OFFSET for value in codec[start:end]] + [MUSIC_END],
              full_noise[start:end])
        for start, end in ranges
    ]


def song_chunks(prefix, codec, seed, context=CONTEXT):
    from .pipeline import initial_noise
    return _chunks_with_supplied_noise(prefix, codec, seed, context,
                                       initial_noise(len(codec), seed))


def synthesize(
    model: AcousticModel,
    prefix: Sequence[int],
    codec: Sequence[int],
    seed: int,
    *,
    steps=32,
    context=CONTEXT,
    cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    noise=None,
    query_chunk_size=None,
) -> np.ndarray:
    """Return finite CPU float32 ``[frames,64]`` acoustic latents.

    Request-local FP32 noise uses the same historical cuts. Supply saved noise
    for exact-input comparisons with other backends or prior RNG policies.
    """
    if not isinstance(model, AcousticModel):
        raise TypeError("synthesize requires the model returned by load_nar")
    steps = _positive_integer(steps, "steps")
    if query_chunk_size is not None:
        query_chunk_size = _positive_integer(query_chunk_size, "query_chunk_size")
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable")

    if noise is None:
        from .pipeline import initial_noise
        noise = initial_noise(len(codec), seed)
    chunks = _chunks_with_supplied_noise(prefix, codec, seed, context, noise)
    expected_frames = sum(len(chunk.noise) for chunk in chunks)

    output: list[np.ndarray] = []
    total_steps = steps * len(chunks)
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        engine = CachedNAR(model, chunk, query_chunk_size=query_chunk_size, cancelled=cancelled)
        try:
            progress = None
            if on_progress is not None:

                def progress(completed, _chunk_total, *, _index=chunk_index):
                    on_progress(_index * steps + completed, total_steps)

            output.append(engine.solve(steps, cancelled, progress))
        finally:
            engine.close()

    result = output[0] if len(output) == 1 else np.concatenate(output, axis=0)
    if result.ndim != 2 or result.shape != (expected_frames, _LATENT_DIM):
        raise RuntimeError("Acoustic synthesis returned an invalid latent shape")
    if not np.isfinite(result).all():
        raise FloatingPointError("Acoustic synthesis produced non-finite latents")
    return result


__all__ = ["AcousticModel", "CachedNAR", "Chunk", "load_nar", "song_chunks", "synthesize"]
