"""MLX Qwen3 autoregressive backend for YuE2.

The learned AR subset is a Qwen3 model. Generation uses fixed-capacity,
append-only BF16 KV caches and projects only the final hidden state onto the
rows allowed by the current protocol phase. Physical cache slots and RoPE
positions are identical here because AR prompts are unpadded; ``cache.offset``
is the next physical slot and the RoPE offset used by MLX-LM.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from importlib.metadata import version
import json
import math
import re
from operator import index
from pathlib import Path
import time
from types import MappingProxyType

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3
from mlx_lm.models.cache import create_attention_mask as _cache_attention_mask

from yue2.protocol import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    MUSIC_END,
    VOCAB_SIZE,
)

from .sampling import RequestRNG, distribution, end_index, guided_logits, local_to_native

PREFILL_CHUNK_SIZE = 1024
_PRECISION_FILES = {
    "bf16": "ar-bf16.safetensors",
    "8bit": "ar-8bit.safetensors",
    "4bit": "ar-4bit.safetensors",
}
_REQUIRED_CONFIG = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
)


class SourceRMSNorm(nn.Module):
    """YuE2 RMSNorm: FP32 variance, then BF16 normalization and weight multiply."""

    def __init__(self, weight: mx.array, eps: float):
        super().__init__()
        self.weight = weight
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        scale = mx.rsqrt(variance + mx.array(self.eps, dtype=mx.float32)).astype(x.dtype)
        normalized = (x * scale).astype(x.dtype)
        return (normalized * self.weight).astype(x.dtype)


class SourceRoPE(nn.Module):
    """Half-split RoPE with the source model's FP32 angles and BF16 rounding."""

    def __init__(self, dims: int, base: float):
        super().__init__()
        if dims % 2:
            raise ValueError("RoPE head dimension must be even")
        self.dims = dims
        self.base = float(base)
        exponent = mx.arange(0, dims, 2, dtype=mx.float32) / mx.array(
            dims, dtype=mx.float32
        )
        inv_freq = 1.0 / mx.power(mx.array(self.base, dtype=mx.float32), exponent)
        # This is a derived FP32 constant, not a checkpoint parameter.
        object.__setattr__(self, "_inv_freq", inv_freq)

    @property
    def inv_freq(self) -> mx.array:
        return self._inv_freq

    def __call__(self, x: mx.array, offset: int | mx.array = 0) -> mx.array:
        length = x.shape[-2]
        positions = mx.arange(length, dtype=mx.float32) + mx.array(
            offset, dtype=mx.float32
        )
        angles = positions[:, None] * self._inv_freq[None, :]
        cosine = mx.cos(angles).astype(x.dtype)
        sine = mx.sin(angles).astype(x.dtype)
        first, second = x[..., : self.dims // 2], x[..., self.dims // 2 :]
        first_cos = (first * cosine).astype(x.dtype)
        second_sin = (second * sine).astype(x.dtype)
        second_cos = (second * cosine).astype(x.dtype)
        first_sin = (first * sine).astype(x.dtype)
        rotated_first = (first_cos - second_sin).astype(x.dtype)
        rotated_second = (second_cos + first_sin).astype(x.dtype)
        return mx.concatenate([rotated_first, rotated_second], axis=-1)


@partial(mx.compile, shapeless=True)
def _source_silu(x: mx.array) -> mx.array:
    """Fused FP32 SiLU with only the final result rounded to the input dtype."""
    value = x.astype(mx.float32)
    return (value * mx.sigmoid(value)).astype(x.dtype)


class SourceMLP(nn.Module):
    """Source SwiGLU: FP32 SiLU opmath, then a source-dtype product."""

    def __init__(self, original: qwen3.MLP):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj

    def __call__(self, x: mx.array) -> mx.array:
        gate = _source_silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj((gate * up).astype(x.dtype))


def _steel_attention_verified(architecture: str, mlx_version: str) -> bool:
    """Only enable BF16 operands for the reviewed pre-NAX Metal kernels.

    MLX v0.32.2 Steel attention uses float MMA tiles for QK, softmax and PV.
    NAX starts at generation 17 (18 for 'p'); unknown/new kernels retain the
    original promotion until reviewed. Real-checkpoint evidence: M3 Max g15s.
    """
    return mlx_version == "0.32.2" and re.fullmatch(r"applegpu_g1[3-6][a-z]", architecture) is not None


@lru_cache(maxsize=1)
def _full_attention_requires_promotion() -> bool:
    try:
        architecture = mx.device_info().get("architecture", "")
    except RuntimeError:
        return True
    return not _steel_attention_verified(architecture, version("mlx"))


def _source_sdpa(query, key, value, *, scale, mask=None):
    """FP32 attention opmath, with BF16 storage outside the operation.

    M5 BF16 full SDPA uses NAX's reduced-precision mixed P@V path even with
    TF32 disabled. FP32 operands select precise Steel SDPA under Lyra's verified
    process policy. Reviewed pre-NAX Steel kernels and the short unmasked vector
    kernel already use FP32 opmath with BF16 operands, avoiding this conversion.
    """
    if (query.shape[2] <= 8 and mask is None) or not _full_attention_requires_promotion():
        return mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )
    return mx.fast.scaled_dot_product_attention(
        query.astype(mx.float32), key.astype(mx.float32), value.astype(mx.float32),
        scale=scale, mask=mask,
    ).astype(query.dtype)


class SourceAttention(nn.Module):
    """Qwen3 projections and cache protocol with source-precision attention."""

    def __init__(self, original: qwen3.Attention):
        super().__init__()
        self.n_heads = original.n_heads
        self.n_kv_heads = original.n_kv_heads
        self.scale = original.scale
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.q_norm = original.q_norm
        self.k_norm = original.k_norm
        self.rope = original.rope

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        query = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        key = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        value = self.v_proj(x).reshape(
            batch, length, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        query, key = self.rope(query, offset=offset), self.rope(key, offset=offset)
        if cache is not None:
            key, value = cache.update_and_fetch(key, value)
        output = _source_sdpa(query, key, value, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class StaticKVCache:
    """One fixed-capacity, append-only BF16 Qwen attention cache.

    Public ``keys`` and ``values`` are views of used tokens with shape
    ``[1, kv_heads, offset, head_dim]``. The private backing arrays retain
    ``max_length`` physical slots and are allocated once. ``offset`` is both
    the number of occupied physical slots and, for YuE2's unpadded AR stream,
    the RoPE position of the next token.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        max_length: int,
        head_dim: int,
        dtype=mx.bfloat16,
    ):
        if batch_size != 1:
            raise ValueError("YuE2 AR caches support exactly one request branch")
        if type(max_length) is not int or max_length < 1:
            raise ValueError("KV cache capacity must be a positive integer")
        self.max_length = max_length
        self.dtype = dtype
        shape = (batch_size, num_kv_heads, max_length, head_dim)
        self._keys = mx.zeros(shape, dtype=dtype)
        self._values = mx.zeros(shape, dtype=dtype)
        self.offset = 0

    @property
    def keys(self) -> mx.array:
        return self._keys[..., : self.offset, :]

    @property
    def values(self) -> mx.array:
        return self._values[..., : self.offset, :]

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        return self.keys, self.values

    @property
    def storage(self) -> tuple[mx.array, mx.array]:
        """Full fixed buffers, exposed separately from used cache views."""
        return self._keys, self._values

    @property
    def nbytes(self) -> int:
        return self._keys.nbytes + self._values.nbytes

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def make_mask(
        self,
        length: int,
        *,
        return_array: bool = False,
        window_size: int | None = None,
    ):
        return _cache_attention_mask(length, self.offset, return_array, window_size)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("KV updates must be matching [B, H, T, D] arrays")
        if keys.shape[0] != self._keys.shape[0] or keys.shape[1] != self._keys.shape[1]:
            raise ValueError("KV update batch/head shape does not match the cache")
        if keys.shape[-1] != self._keys.shape[-1]:
            raise ValueError("KV update head dimension does not match the cache")
        start = self.offset
        end = start + keys.shape[-2]
        if end > self.max_length:
            raise ValueError(
                f"KV cache capacity {self.max_length} exceeded by {end}; "
                "generation was not shortened"
            )
        self._keys[..., start:end, :] = keys.astype(self.dtype)
        self._values[..., start:end, :] = values.astype(self.dtype)
        self.offset = end
        return self.keys, self.values

    def reset(self) -> None:
        self.offset = 0

    def close(self) -> None:
        self.offset = 0
        self._keys = None
        self._values = None


class ARModel(qwen3.Model):
    """Qwen3-compatible model with YuE2's source arithmetic and row projection."""

    def __init__(self, args: qwen3.ModelArgs, source_config: Mapping[str, object]):
        super().__init__(args)
        object.__setattr__(self, "_source_config", MappingProxyType(dict(source_config)))
        object.__setattr__(self, "_precision", "bf16")

    @property
    def config(self) -> Mapping[str, object]:
        """Original generator config; architectural runtime fields are in ``args``."""
        return self._source_config

    @property
    def precision(self) -> str:
        return self._precision

    def project(self, hidden: mx.array, phase: str) -> mx.array:
        return project_logits(self, hidden, phase)

    def make_cache(self, capacity: int) -> list[StaticKVCache]:
        return make_cache(self, capacity)


def _as_config(config) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("config must be a mapping or expose to_dict()")


def _model_args(config: Mapping[str, object]) -> qwen3.ModelArgs:
    missing = [name for name in _REQUIRED_CONFIG if name not in config]
    if missing:
        raise ValueError(f"Generator config is missing AR fields: {', '.join(missing)}")
    if config.get("tie_word_embeddings", False) is not False:
        raise ValueError("YuE2 requires untied input and output embeddings")
    if config.get("rope_scaling") not in (None, {}):
        raise ValueError("YuE2 uses fixed base RoPE without scaling")

    values = {name: config[name] for name in _REQUIRED_CONFIG}
    values.update(
        model_type="qwen3",
        max_position_embeddings=config.get("max_position_embeddings", CONTEXT),
        tie_word_embeddings=False,
        rope_scaling=None,
    )
    integer_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )
    for name in integer_fields:
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f"Generator config field {name} must be a positive integer")
    if values["num_attention_heads"] % values["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if not math.isfinite(float(values["rms_norm_eps"])) or values["rms_norm_eps"] <= 0:
        raise ValueError("rms_norm_eps must be finite and positive")
    if not math.isfinite(float(values["rope_theta"])) or values["rope_theta"] <= 0:
        raise ValueError("rope_theta must be finite and positive")
    return qwen3.ModelArgs.from_dict(values)


def _install_source_math(model: ARModel) -> None:
    args = model.args
    model.model.norm = SourceRMSNorm(model.model.norm.weight, args.rms_norm_eps)
    for layer in model.model.layers:
        layer.input_layernorm = SourceRMSNorm(
            layer.input_layernorm.weight, args.rms_norm_eps
        )
        layer.post_attention_layernorm = SourceRMSNorm(
            layer.post_attention_layernorm.weight, args.rms_norm_eps
        )
        attention = layer.self_attn
        attention.q_norm = SourceRMSNorm(attention.q_norm.weight, args.rms_norm_eps)
        attention.k_norm = SourceRMSNorm(attention.k_norm.weight, args.rms_norm_eps)
        attention.rope = SourceRoPE(args.head_dim, args.rope_theta)
        layer.self_attn = SourceAttention(attention)
        layer.mlp = SourceMLP(layer.mlp)


def build_ar(config) -> ARModel:
    """Build the exact AR subset as an MLX-LM Qwen3-compatible BF16 model."""
    source_config = _as_config(config)
    args = _model_args(source_config)
    model = ARModel(args, source_config)
    _install_source_math(model)
    model.set_dtype(mx.bfloat16)
    model.eval()
    return model


def _validate_protocol_config(config: Mapping[str, object]) -> None:
    expected = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": CONTEXT,
        "rope_theta": 1_000_000.0,
    }
    for name, wanted in expected.items():
        actual = config.get(name, CONTEXT if name == "max_position_embeddings" else None)
        if actual != wanted:
            raise ValueError(f"Incompatible YuE2 config: {name}={actual!r}, expected {wanted!r}")
    if config.get("tie_word_embeddings", False) is not False:
        raise ValueError("Incompatible YuE2 config: embeddings must be untied")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("Incompatible YuE2 config: hidden_act must be silu")


def _read_manifest(directory: Path, verify: bool) -> dict:
    if verify:
        from .conversion import verify_conversion

        manifest = verify_conversion(directory)
    else:
        with (directory / "conversion.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("conversion.json must be a schema-1 manifest")
    return manifest


def load_ar(directory, *, precision: str = "bf16", verify: bool = True) -> ARModel:
    """Strictly load a converted AR model in BF16, affine 8-bit, or affine 4-bit."""
    if precision not in _PRECISION_FILES:
        raise ValueError("precision must be 'bf16', '8bit', or '4bit'")
    directory = Path(directory)
    manifest = _read_manifest(directory, verify)
    precisions = manifest.get("precisions")
    if not isinstance(precisions, dict) or not isinstance(precisions.get(precision), dict):
        raise FileNotFoundError(f"Converted artifact does not contain {precision} AR weights")
    entry = precisions[precision]
    expected_file = _PRECISION_FILES[precision]
    if entry.get("ar") != expected_file:
        raise ValueError(
            f"Manifest {precision} AR file must be {expected_file!r}, got {entry.get('ar')!r}"
        )

    with (directory / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    _validate_protocol_config(config)
    model = build_ar(config)

    if precision == "bf16":
        if entry.get("dtype") != "bfloat16":
            raise ValueError("BF16 manifest entry must declare dtype='bfloat16'")
    else:
        bits = 8 if precision == "8bit" else 4
        settings = {
            "bits": bits,
            "group_size": 64,
            "mode": "affine",
            "linear_only": True,
        }
        for name, wanted in settings.items():
            if entry.get(name) != wanted:
                raise ValueError(
                    f"Manifest {precision} setting {name}={entry.get(name)!r}, "
                    f"expected {wanted!r}"
                )
        nn.quantize(
            model,
            group_size=64,
            bits=bits,
            mode="affine",
            class_predicate=lambda _path, module: isinstance(module, nn.Linear),
        )

    weight_path = directory / expected_file
    weights = mx.load(str(weight_path))
    if not isinstance(weights, dict):
        raise ValueError(f"{expected_file} did not contain named tensors")
    wrong_dtype = [
        name
        for name, value in weights.items()
        if mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.bfloat16
    ]
    if wrong_dtype:
        preview = ", ".join(sorted(wrong_dtype)[:5])
        raise ValueError(f"AR floating tensors must be BF16; incompatible tensors: {preview}")
    model.load_weights(list(weights.items()), strict=True)
    del weights
    mx.eval(model.parameters())
    object.__setattr__(model, "_precision", precision)
    return model


def make_cache(model: ARModel, capacity: int) -> list[StaticKVCache]:
    """Create per-layer fixed BF16 caches for one unpadded AR branch.

    Cache tensors exposed as ``cache.keys``/``cache.values`` always have used
    shape ``[1, num_key_value_heads, cache.offset, head_dim]``, regardless of
    whether model weights are BF16, 8-bit, or 4-bit.
    """
    if type(capacity) is not int or not 1 <= capacity <= model.args.max_position_embeddings:
        raise ValueError(
            f"Cache capacity must be an integer in [1, {model.args.max_position_embeddings}]"
        )
    return [
        StaticKVCache(
            batch_size=1,
            num_kv_heads=model.args.num_key_value_heads,
            max_length=capacity,
            head_dim=model.args.head_dim,
            dtype=mx.bfloat16,
        )
        for _ in range(model.args.num_hidden_layers)
    ]


def _native_tokens(token_ids: Sequence[int], vocab_size: int) -> tuple[int, ...]:
    try:
        tokens = tuple(index(token) for token in token_ids)
    except TypeError as error:
        raise TypeError("Token IDs must be integers") from error
    if not tokens:
        raise ValueError("AR prefix must contain at least one token")
    if any(token < 0 or token >= vocab_size for token in tokens):
        raise ValueError(f"Token IDs must be in [0, {vocab_size})")
    return tokens


def _linear_rows(layer: nn.Module, hidden: mx.array, start: int, stop: int) -> mx.array:
    if isinstance(layer, nn.QuantizedLinear):
        output = mx.quantized_matmul(
            hidden,
            layer.weight[start:stop],
            scales=layer.scales[start:stop],
            biases=layer.biases[start:stop] if layer.get("biases") is not None else None,
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )
    elif isinstance(layer, nn.Linear):
        output = hidden @ layer.weight[start:stop].T
    else:
        raise TypeError(f"Unsupported output projection type {type(layer).__name__}")
    if layer.get("bias") is not None:
        output = output + layer.bias[start:stop]
    return output


def project_logits(model: ARModel, hidden: mx.array, phase: str) -> mx.array:
    """Project hidden states onto exactly the rows allowed by ``phase``.

    ABC local rows are native ``[0:EOD)`` followed by ``ABC_END``. Semantic
    local rows are the contiguous native slice ``[MUSIC_END:CODEC_OFFSET +
    CODEC_SIZE)`` (end token first, then 32768 codec IDs).
    """
    if hidden.shape[-1] != model.args.hidden_size:
        raise ValueError("Hidden-state width does not match the AR model")
    if phase == "abc":
        if model.args.vocab_size <= ABC_END:
            raise ValueError("AR vocabulary does not contain the ABC end row")
        content = _linear_rows(model.lm_head, hidden, 0, EOD)
        end = _linear_rows(model.lm_head, hidden, ABC_END, ABC_END + 1)
        logits = mx.concatenate([content, end], axis=-1)
    elif phase == "semantic":
        stop = CODEC_OFFSET + CODEC_SIZE
        if model.args.vocab_size < stop:
            raise ValueError("AR vocabulary does not contain all semantic rows")
        logits = _linear_rows(model.lm_head, hidden, MUSIC_END, stop)
    else:
        raise ValueError("phase must be 'abc' or 'semantic'")
    # Source model logits are BF16 even when the weight matrices are quantized.
    return logits.astype(mx.bfloat16)


def prefill(
    model: ARModel,
    token_ids: Sequence[int],
    *,
    capacity: int,
    phase: str,
    chunk_size: int = PREFILL_CHUNK_SIZE,
    cancelled=None,
) -> tuple[mx.array, list[StaticKVCache]]:
    """Prefill fixed caches and return phase-local logits for the final token.

    Intermediate chunks run only the backbone. Consequently no chunk ever
    materializes all-position vocabulary logits; only one final hidden state is
    sent through the phase-local output projection.
    """
    tokens = _native_tokens(token_ids, model.args.vocab_size)
    if capacity < len(tokens):
        raise ValueError(f"Cache capacity {capacity} is smaller than prefix length {len(tokens)}")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("prefill chunk_size must be a positive integer")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")
    caches = make_cache(model, capacity)
    try:
        inputs = mx.array([tokens], dtype=mx.int32)
        hidden = None
        for start in range(0, len(tokens), chunk_size):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during prefill")
            stop = min(start + chunk_size, len(tokens))
            hidden = model.model(inputs[:, start:stop], cache=caches)
            if stop < len(tokens):
                mx.eval([cache.state for cache in caches])
                del hidden
                hidden = None
                mx.clear_cache()
        assert hidden is not None
        logits = project_logits(model, hidden[:, -1, :], phase)
        return logits, caches
    except BaseException:
        _close_caches(caches)
        raise


def _close_caches(caches: list[StaticKVCache] | None) -> None:
    if caches is not None:
        for cache in caches:
            cache.close()


def generate_tokens(
    model: ARModel,
    prefix: Sequence[int],
    sampling,
    seed: int,
    phase: str,
    negative: Sequence[int] | None = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
    cancelled=None,
    on_token=None,
) -> tuple[list[int], dict, bool]:
    """Generate native content IDs with bounded caches and local MLX sampling.

    The request-local key is recreated from ``seed`` for each call, matching the
    phase reset in the reference implementation without touching MLX's global
    RNG. Sampling remains on device. Observing EOS and invoking ``on_token``
    necessarily resolve one scalar per token; there are no additional decode-
    loop synchronizations.
    """
    if phase not in {"abc", "semantic"}:
        raise ValueError("phase must be 'abc' or 'semantic'")
    prefix = _native_tokens(prefix, model.args.vocab_size)
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1.0 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    negative_tokens = None
    if negative is not None:
        negative_tokens = _native_tokens(negative, model.args.vocab_size)
        if len(negative_tokens) + sampling.max_tokens > CONTEXT:
            raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")

    rng = RequestRNG(seed)
    positive_cache = None
    negative_cache = None
    mx.synchronize()
    started = time.perf_counter()
    try:
        conditional, positive_cache = prefill(
            model,
            prefix,
            capacity=len(prefix) + sampling.max_tokens,
            phase=phase,
            cancelled=cancelled,
        )
        unconditional = None
        if cfg_scale != 1.0:
            unconditional, negative_cache = prefill(
                model,
                negative_tokens,
                capacity=len(negative_tokens) + sampling.max_tokens,
                phase=phase,
                cancelled=cancelled,
            )
        if unconditional is None:
            mx.eval(conditional)
        else:
            mx.eval(conditional, unconditional)
        mx.synchronize()
        prefill_seconds = time.perf_counter() - started

        history: list[int] = []
        first = None
        eos = False
        end = end_index(phase)
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            logits = guided_logits(conditional, unconditional, cfg_scale)
            scores = distribution(logits, sampling, history, step, phase, legacy_off)
            if sampling.temperature == 0:
                next_local = mx.argmax(scores, axis=-1)
            else:
                next_local = rng.categorical(scores)
            local_token = int(next_local.item())
            token = local_to_native(local_token, phase)
            if first is None:
                first = time.perf_counter() - started
            if on_token is not None:
                on_token(phase, token)
            if local_token == end:
                eos = True
                break
            history.append(token)

            if step + 1 < sampling.max_tokens:
                input_id = mx.array([[token]], dtype=mx.int32)
                hidden = model.model(input_id, cache=positive_cache)
                conditional = project_logits(model, hidden[:, -1, :], phase)
                if negative_cache is not None:
                    negative_hidden = model.model(input_id, cache=negative_cache)
                    unconditional = project_logits(model, negative_hidden[:, -1, :], phase)

        mx.synchronize()
        seconds = time.perf_counter() - started
        count = len(history) + int(eos)
        timing = {
            "seconds": seconds,
            "prefill_seconds": prefill_seconds,
            "ttft_seconds": first,
            "output_tokens": count,
            "content_tokens": len(history),
            "output_tps": count / seconds,
            "prefix_tokens": len(prefix),
            "cfg_branches": 1 if cfg_scale == 1.0 else 2,
            "execution": "eager",
            "attention": "sdpa",
            "backend": "mlx",
        }
        return history, timing, not eos
    finally:
        _close_caches(positive_cache)
        _close_caches(negative_cache)


__all__ = [
    "ARModel",
    "PREFILL_CHUNK_SIZE",
    "SourceMLP",
    "SourceRMSNorm",
    "SourceRoPE",
    "StaticKVCache",
    "build_ar",
    "generate_tokens",
    "load_ar",
    "make_cache",
    "prefill",
    "project_logits",
]
