"""Device-local YuE2 sampling with the native protocol's exact filters.

Production sampling operates on phase-local logits rather than the 184704-row
head: ABC rows are ``[0:EOD) + [ABC_END]`` and semantic rows are
``[MUSIC_END:LATENT_START)``. Token histories and returned tokens remain in the
checkpoint's native ID space.
"""
from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from yue2.protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END

ABC_SIZE = EOD + 1
SEMANTIC_SIZE = CODEC_SIZE + 1


def _phase_size(phase: str) -> int:
    if phase == "abc":
        return ABC_SIZE
    if phase == "semantic":
        return SEMANTIC_SIZE
    raise ValueError("phase must be 'abc' or 'semantic'")


def end_index(phase: str) -> int:
    """Return the phase-local end-token index."""
    _phase_size(phase)
    return EOD if phase == "abc" else 0


def local_to_native(token: int, phase: str) -> int:
    """Map a phase-local sampled index back to its checkpoint-native ID."""
    size = _phase_size(phase)
    if not 0 <= token < size:
        raise ValueError(f"Local {phase} token {token} is outside [0, {size})")
    if phase == "abc":
        return token if token < EOD else ABC_END
    return MUSIC_END + token


def native_to_local(token: int, phase: str) -> int | None:
    """Map an allowed native ID to a local row, or return ``None`` if disallowed."""
    _phase_size(phase)
    if phase == "abc":
        if 0 <= token < EOD:
            return token
        return EOD if token == ABC_END else None
    if token == MUSIC_END:
        return 0
    if CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE:
        return token - MUSIC_END
    return None


def window_penalty(logits: mx.array, recent_ids: Sequence[int], penalty: float) -> mx.array:
    """Apply YuE2's frequency-powered repetition penalty to local row IDs.

    Repeated IDs are accumulated rather than updated once. This is the material
    distinction from MLX-LM's ordinary repetition processor.
    """
    if penalty == 1.0 or not recent_ids:
        return logits
    ids = mx.array(recent_ids, dtype=mx.int32)
    frequency = mx.zeros((logits.shape[-1],), dtype=logits.dtype)
    frequency = frequency.at[ids].add(mx.ones(ids.shape, dtype=logits.dtype))
    # PyTorch wrapped Python scalars use FP32 opmath even with BF16 tensors.
    # Round the power result, not the scalar base, to the source tensor dtype.
    base = mx.array(penalty, dtype=mx.float32)
    alpha = mx.power(base, frequency.astype(mx.float32)).astype(logits.dtype)
    multiplied = (logits * alpha).astype(logits.dtype)
    divided = (logits / alpha).astype(logits.dtype)
    return mx.where(logits < 0, multiplied, divided)


def distribution(
    logits: mx.array,
    sampling,
    history: Sequence[int],
    step: int,
    phase: str,
    legacy_off: bool = False,
) -> mx.array:
    """Apply native min-length, repetition, temperature, top-k and top-p rules.

    ``logits`` must already contain exactly the allowed phase rows. Symbolic and
    ordinary semantic sampling upcast to FP32. Historical ``off`` sampling keeps
    BF16 throughout. Top-k masks by the kth *value* so ties survive, and nucleus
    filtering uses the upstream shifted cumulative test.
    """
    expected = _phase_size(phase)
    if logits.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} projected {phase} logits, got {logits.shape[-1]}"
        )
    scores = logits.astype(logits.dtype if legacy_off else mx.float32)
    end = end_index(phase)
    if step < sampling.min_tokens:
        positions = mx.arange(expected)
        scores = mx.where(positions == end, mx.array(-float("inf"), scores.dtype), scores)

    recent_native = history[-sampling.penalty_window :]
    recent_local = [
        local
        for token in recent_native
        if (local := native_to_local(int(token), phase)) is not None
    ]
    scores = window_penalty(scores, recent_local, sampling.repetition_penalty)

    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        temperature = mx.array(sampling.temperature, dtype=mx.float32)
        scores = (scores.astype(mx.float32) / temperature).astype(scores.dtype)

    k = min(sampling.top_k, expected)
    threshold = mx.min(mx.topk(scores, k, axis=-1), axis=-1, keepdims=True)
    scores = mx.where(scores < threshold, mx.array(-float("inf"), scores.dtype), scores)

    if sampling.top_p < 1:
        indices = mx.argsort(-scores, axis=-1)
        values = mx.take_along_axis(scores, indices, axis=-1)
        probabilities = mx.softmax(values, axis=-1, precise=True)
        preceding_mass = (mx.cumsum(probabilities, axis=-1) - probabilities).astype(scores.dtype)
        removed = preceding_mass.astype(mx.float32) > mx.array(sampling.top_p, dtype=mx.float32)
        minimum = 3 if legacy_off else 1
        removed = removed & (mx.arange(expected) >= minimum)
        values = mx.where(removed, mx.array(-float("inf"), values.dtype), values)
        # Every destination occurs exactly once; using the sorted values as the
        # initial array therefore reproduces torch.scatter's result.
        scores = mx.put_along_axis(values, indices, values, axis=-1)
    return scores


def guided_logits(
    conditional: mx.array,
    unconditional: mx.array | None,
    cfg_scale: float,
) -> mx.array:
    """Apply historical CFG with an explicit rounding point after each BF16 op."""
    if unconditional is None:
        return conditional
    if conditional.shape != unconditional.shape:
        raise ValueError("Conditional and unconditional logits must have the same shape")
    if conditional.dtype != unconditional.dtype:
        raise ValueError("Conditional and unconditional logits must have the same dtype")
    dtype = conditional.dtype
    delta = (conditional - unconditional).astype(dtype)
    scaled = (delta.astype(mx.float32) * mx.array(cfg_scale, dtype=mx.float32)).astype(dtype)
    return (unconditional + scaled).astype(dtype)


class RequestRNG:
    """A request-local MLX PRNG which never reads or mutates global RNG state."""

    __slots__ = ("_key",)

    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        # MLX keys are two uint32 words. Construct both explicitly so seeds in
        # [2**32, 2**63) cannot alias their low-word counterpart; for ordinary
        # 32-bit seeds this is identical to the conventional [0, seed] key.
        self._key = mx.array([seed >> 32, seed & 0xFFFFFFFF], dtype=mx.uint32)

    @property
    def key(self) -> mx.array:
        """Current key, exposed for deterministic sampling probes."""
        return self._key

    def categorical(self, logits: mx.array) -> mx.array:
        keys = mx.random.split(self._key, num=2)
        self._key = keys[0]
        # Preserve the final source-dtype softmax, including historical BF16
        # probability rounding, before using MLX's independent random stream.
        probabilities = mx.softmax(logits, axis=-1, precise=True)
        log_probabilities = mx.log(probabilities.astype(mx.float32))
        return mx.random.categorical(log_probabilities, axis=-1, key=keys[1])


__all__ = [
    "ABC_SIZE",
    "SEMANTIC_SIZE",
    "RequestRNG",
    "distribution",
    "end_index",
    "guided_logits",
    "local_to_native",
    "native_to_local",
    "window_penalty",
]
