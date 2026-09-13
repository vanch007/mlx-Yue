"""Retain FP32 probability accumulation while selecting verified Steel kernels."""
import mlx.core as mx
import numpy as np
import pytest

from lyra import ar
from lyra.nar import _attention


@pytest.mark.parametrize("architecture,version,expected", [
    ("applegpu_g15s", "0.32.2", True),
    ("applegpu_g16s", "0.32.2", True),
    ("applegpu_g13g", "0.32.2", True),
    ("applegpu_g17s", "0.32.2", False),
    ("applegpu_g18p", "0.32.2", False),
    ("applegpu_g15s", "0.33.0", False),
    ("unknown", "0.32.2", False),
])
def test_unreviewed_kernels_keep_original_promotion(architecture, version, expected):
    assert ar._steel_attention_verified(architecture, version) is expected


@pytest.mark.parametrize("causal", [False, True])
def test_steel_bf16_inputs_match_original_promoted_attention(monkeypatch, causal):
    if ar._full_attention_requires_promotion():
        pytest.skip("Native BF16 operand equivalence requires a verified Steel GPU")
    rng = np.random.default_rng(123)
    q = mx.array(rng.normal(size=(1, 4, 35, 128)), dtype=mx.bfloat16)
    k = mx.array(rng.normal(size=(1, 2, 35 if causal else 57, 128)), dtype=mx.bfloat16)
    v = mx.array(rng.normal(size=k.shape), dtype=mx.bfloat16)
    actual = _attention(q, k, v, causal=causal, query_chunk_size=16)
    mx.eval(actual)
    monkeypatch.setattr(ar, "_full_attention_requires_promotion", lambda: True)
    expected = _attention(q, k, v, causal=causal, query_chunk_size=16)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),
                                  np.array(expected.astype(mx.float32)))
