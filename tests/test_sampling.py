"""Numerical filters, not sampled token equality across different RNG backends."""
from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch

from lyra.ar import project_logits
from lyra.sampling import RequestRNG, distribution, guided_logits, local_to_native
from yue2.protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END, Sampling, VOCAB_SIZE
from yue2.sampling import distribution as reference_distribution


def numpy_mx(value):
    return np.asarray(value.astype(mx.float32))


def projected(value, phase):
    if phase == "abc":
        return np.concatenate((value[..., :EOD], value[..., ABC_END:ABC_END + 1]), axis=-1)
    return value[..., MUSIC_END:CODEC_OFFSET + 32768]


def compare_filter(logits, config, history, step=0, phase="semantic", off=False):
    source = torch.from_numpy(logits).to(device="mps", dtype=torch.bfloat16)
    expected = reference_distribution(source, config, history, step, phase, off).float().cpu().numpy()
    compact = projected(source.float().cpu().numpy(), phase)
    actual = numpy_mx(distribution(mx.array(compact, dtype=mx.bfloat16), config, history, step, phase, off))
    expected = projected(expected, phase)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=1e-6)
    return actual


def test_topk_preserves_all_cutoff_ties():
    logits = np.full((1, VOCAB_SIZE), -20, dtype=np.float32)
    logits[0, CODEC_OFFSET:CODEC_OFFSET + 4] = [5, 4, 4, 3]
    actual = compare_filter(logits, Sampling(top_k=2, top_p=1, min_tokens=0, repetition_penalty=1), [])
    assert np.flatnonzero(np.isfinite(actual[0])).tolist() == [1, 2, 3]


@pytest.mark.parametrize("off", [False, True])
def test_shifted_nucleus_and_mode_specific_minimum(off):
    logits = np.full((1, VOCAB_SIZE), -20, dtype=np.float32)
    logits[0, CODEC_OFFSET:CODEC_OFFSET + 4] = [5, 4, 3, 2]
    actual = compare_filter(logits, Sampling(top_k=4, top_p=.01, min_tokens=0, repetition_penalty=1), [], off=off)
    assert np.flatnonzero(np.isfinite(actual[0])).tolist() == ([1, 2, 3] if off else [1])


def test_off_nucleus_compares_against_unrounded_python_threshold():
    logits = np.full((1, VOCAB_SIZE), -np.inf, dtype=np.float32)
    logits[0, CODEC_OFFSET:CODEC_OFFSET + 4] = 0
    config = Sampling(top_k=4, top_p=.749, min_tokens=0, repetition_penalty=1)
    source = torch.from_numpy(logits).to("mps", torch.bfloat16)
    expected = reference_distribution(source, config, [], 0, "semantic", True)
    actual = distribution(mx.array(projected(logits, "semantic"), dtype=mx.bfloat16),
                          config, [], 0, "semantic", True)
    # Equal logits can have backend-specific sort order, but not a different
    # retained mass/count at the shifted-CDF boundary above the keep-three floor.
    assert int(torch.isfinite(expected).sum().item()) == 3
    assert int(np.isfinite(numpy_mx(actual)).sum()) == 3


@pytest.mark.parametrize("phase,content,end", [
    ("abc", EOD - 1, ABC_END),
    ("semantic", CODEC_OFFSET + CODEC_SIZE - 1, MUSIC_END),
])
def test_projected_sampling_preserves_native_boundaries_and_minimum(phase, content, end):
    weights = np.zeros((VOCAB_SIZE, 1), dtype=np.float32)
    weights[content, 0], weights[end, 0] = 3, 4
    # Neither the excluded EOD row nor trailing vocabulary rows may win.
    weights[EOD, 0], weights[-1, 0] = 100, 101
    head = nn.Linear(1, VOCAB_SIZE, bias=False)
    head.weight = mx.array(weights, dtype=mx.bfloat16)
    model = SimpleNamespace(args=SimpleNamespace(hidden_size=1, vocab_size=VOCAB_SIZE),
                            lm_head=head)
    compact = project_logits(model, mx.array([[2]], dtype=mx.bfloat16), phase)
    source = torch.from_numpy((weights * 2).T.copy()).to("mps", torch.bfloat16)
    config = Sampling(temperature=0, top_k=1, top_p=.01, min_tokens=1)
    for step, expected_native in ((0, content), (1, end)):
        expected = reference_distribution(source, config, [], step, phase, False)
        actual = distribution(compact, config, [], step, phase)
        native = local_to_native(int(mx.argmax(actual).item()), phase)
        assert native == int(expected.argmax().item()) == expected_native
        # Greedy decoding bypasses top-k/nucleus without dropping valid rows.
        np.testing.assert_array_equal(
            np.isfinite(numpy_mx(actual)),
            projected(torch.isfinite(expected).cpu().numpy(), phase),
        )


@pytest.mark.parametrize("off", [False, True])
def test_repetition_counts_expire_with_window_and_preserve_sign(off):
    logits = np.full((1, VOCAB_SIZE), -20, dtype=np.float32)
    logits[0, CODEC_OFFSET:CODEC_OFFSET + 3] = [4.5, -4.5, 3]
    history = [CODEC_OFFSET + 2] * 5 + [CODEC_OFFSET, CODEC_OFFSET, CODEC_OFFSET + 1]
    compare_filter(logits, Sampling(temperature=.7, top_p=1, top_k=20, min_tokens=0,
                                   repetition_penalty=1.2, penalty_window=3), history, off=off)


def test_fractional_cfg_uses_historical_bf16_scalar_arithmetic():
    conditional = torch.tensor([[4.5, -3.25, 0.00390625, 18.5]], device="mps", dtype=torch.bfloat16)
    negative = torch.tensor([[-1.25, 2.75, -0.015625, 17.0]], device="mps", dtype=torch.bfloat16)
    expected = (negative + 1.01 * (conditional - negative)).float().cpu().numpy()
    actual = guided_logits(mx.array(conditional.float().cpu().numpy(), dtype=mx.bfloat16),
                           mx.array(negative.float().cpu().numpy(), dtype=mx.bfloat16), 1.01)
    np.testing.assert_array_equal(numpy_mx(actual), expected)


def test_request_rng_is_reproducible_interleaved_and_uses_high_seed_bits():
    logits = mx.zeros((128, 17), dtype=mx.float32)
    first, repeat, high = RequestRNG(17), RequestRNG(17), RequestRNG(2**40 + 17)
    initial_global = [np.asarray(value).copy() for value in mx.random.state]
    one = np.asarray(first.categorical(logits))
    different = np.asarray(high.categorical(logits))
    repeated = np.asarray(repeat.categorical(logits))
    np.testing.assert_array_equal(one, repeated)
    assert not np.array_equal(one, different)
    for expected, actual in zip(initial_global, mx.random.state):
        np.testing.assert_array_equal(expected, np.asarray(actual))
