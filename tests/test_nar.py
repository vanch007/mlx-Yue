"""Defend source solver arithmetic and original, non-streaming acoustic cuts."""
import mlx.core as mx
import numpy as np
import pytest
import torch

from lyra.nar import AcousticModel, CachedNAR, _attention, _chunks_with_supplied_noise
from yue2.protocol import CODEC_OFFSET, MUSIC_END


class AnalyticFlow(CachedNAR):
    """The ODE dx/dt=x isolates midpoint integration from learned weights."""

    def __init__(self, initial):
        self._closed = False
        self._initial_state = mx.array(initial, dtype=mx.bfloat16)
        self.nar_length = len(initial) + 2

    def velocity(self, state, raw_t):
        return state


@pytest.mark.parametrize("steps", [7, 32])
def test_midpoint_matches_source_bf16_solver_with_fractional_step_size(steps):
    initial = np.linspace(-19.7, 21.3, 128, dtype=np.float32).reshape(2, 64)
    expected = torch.from_numpy(initial).to("mps", torch.bfloat16)
    for step in range(steps):
        midpoint = expected - expected * (1 / steps / 2)
        expected = expected - midpoint * (1 / steps)
    flow = AnalyticFlow(initial)
    progress = []
    actual = flow.solve(steps=steps, on_progress=lambda completed, total: progress.append((completed, total)))
    np.testing.assert_array_equal(actual, expected.float().cpu().numpy())
    assert progress == [(step, steps) for step in range(1, steps + 1)]


def test_supplied_noise_uses_original_prefix_and_local_codec_at_every_cut():
    prefix = [10, 11, 12]
    codec = list(range(15))
    noise = np.arange(15 * 64, dtype=np.float32).reshape(15, 64)
    chunks = _chunks_with_supplied_noise(prefix, codec, 42, 20, noise)
    ranges = [(0, 7), (7, 14), (14, 15)]
    assert len(chunks) == 3
    for chunk, (start, end) in zip(chunks, ranges):
        assert chunk.ar_tokens == prefix + [CODEC_OFFSET + token for token in codec[start:end]] + [MUSIC_END]
        np.testing.assert_array_equal(chunk.noise, noise[start:end])


@pytest.mark.parametrize("causal", [False, True])
def test_query_tiling_retains_complete_attention_visibility(causal):
    generator = np.random.default_rng(42)
    q = mx.array(generator.normal(size=(1, 4, 9, 8)).astype(np.float32))
    k = mx.array(generator.normal(size=(1, 2, 9 if causal else 13, 8)).astype(np.float32))
    v = mx.array(generator.normal(size=k.shape).astype(np.float32))
    expected = _attention(q, k, v, causal=causal, query_chunk_size=None)
    tiled = _attention(q, k, v, causal=causal, query_chunk_size=3)
    np.testing.assert_allclose(np.asarray(tiled), np.asarray(expected), rtol=2e-5, atol=2e-6)


def test_attention_preserves_fp32_probabilities_before_bf16_output():
    # Nearly cancelling values expose rounding of softmax probabilities in the
    # M5 NAX mixed-precision P@V path, even with BF16 Q/K/V and TF32 disabled.
    query = np.zeros((1, 1, 9, 128), dtype=np.float32)
    query[..., 0] = 1
    key = np.zeros((1, 1, 12, 128), dtype=np.float32)
    key[..., 0] = np.tile([0, 11.3125, 22.625], 4)
    value = np.broadcast_to(
        np.tile([1, 1, -0.5], 4)[None, None, :, None], key.shape
    ).astype(np.float32).copy()
    # Each of the three scores occurs four times. Derive the reference with
    # NumPy FP64, then round only the final output to BF16. Torch 2.13 MPS SDPA
    # changes this cancellation case, so it cannot define the precision contract.
    probabilities = np.exp(np.array([0, 11.3125, 22.625]) / np.sqrt(128))
    probabilities /= probabilities.sum()
    expected = torch.tensor(float(probabilities @ [1, 1, -0.5])).bfloat16().float().item()
    rounded_probabilities = torch.from_numpy(probabilities).bfloat16().double().numpy()
    assert abs(float(rounded_probabilities @ [1, 1, -0.5]) - expected) > 1e-4
    actual = _attention(
        *(mx.array(array, dtype=mx.bfloat16) for array in (query, key, value)),
        causal=False,
        query_chunk_size=256,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.full(query.shape, expected, dtype=np.float32)
    )


def test_cancelled_solver_can_restart_from_original_noise():
    initial = np.linspace(-2, 2, 128, dtype=np.float32).reshape(2, 64)
    flow = AnalyticFlow(initial)
    with pytest.raises(InterruptedError):
        flow.solve(cancelled=lambda: True)
    np.testing.assert_array_equal(flow.solve(), AnalyticFlow(initial).solve())


def test_timestep_shift_preserves_sigmoid_opmath_and_fractional_coefficient():
    shift = 1.01
    model = AcousticModel({
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
        "rms_norm_eps": 1e-6, "rope_theta": 1e6, "latent_dim": 64,
        "max_latent_frames": 16, "timestep_shift": shift,
    }, ar_model=None)
    values = [-16, -5, -2.71875, -.1875, 0, .1875, 2.71875, 5, 16]
    source = torch.sigmoid(torch.tensor(values, device="mps", dtype=torch.bfloat16))
    expected = (shift * source / (1 + (shift - 1) * source)).float().cpu().numpy()
    actual = np.array([float(model.shift_timestep(value)) for value in values], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
