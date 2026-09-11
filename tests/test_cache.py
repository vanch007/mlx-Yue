"""Append-only physical slots and causal visibility must survive chunked prefill."""
import mlx.core as mx
import numpy as np
import pytest

from lyra.ar import StaticKVCache


def test_append_preserves_history_and_overflow_does_not_mutate_cache():
    cache = StaticKVCache(batch_size=1, num_kv_heads=2, max_length=5, head_dim=4)
    first = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4).astype(mx.bfloat16)
    second = mx.full((1, 2, 2, 4), 42, dtype=mx.bfloat16)
    cache.update_and_fetch(first, first + 1)
    mx.eval(cache.state)
    cache.update_and_fetch(second, second + 1)
    expected = np.concatenate((np.asarray(first.astype(mx.float32)), np.asarray(second.astype(mx.float32))), axis=2)
    np.testing.assert_array_equal(np.asarray(cache.keys.astype(mx.float32)), expected)
    with pytest.raises(ValueError):
        cache.update_and_fetch(second, second)
    assert cache.offset == 5
    np.testing.assert_array_equal(np.asarray(cache.keys.astype(mx.float32)), expected)


def test_prefill_chunk_mask_uses_absolute_physical_positions():
    cache = StaticKVCache(batch_size=1, num_kv_heads=2, max_length=8, head_dim=4)
    old = mx.zeros((1, 2, 3, 4), dtype=mx.bfloat16)
    cache.update_and_fetch(old, old)
    mask = np.asarray(cache.make_mask(2, return_array=True))
    np.testing.assert_array_equal(mask, np.array([[True, True, True, True, False],
                                                [True, True, True, True, True]]))


def test_reset_does_not_expose_stale_tail_positions():
    cache = StaticKVCache(batch_size=1, num_kv_heads=1, max_length=8, head_dim=4)
    old = mx.full((1, 1, 7, 4), 12, dtype=mx.bfloat16)
    cache.update_and_fetch(old, old)
    mx.eval(cache.state)
    cache.reset()
    new = mx.full((1, 1, 2, 4), 3, dtype=mx.bfloat16)
    cache.update_and_fetch(new, new)
    assert cache.offset == 2
    np.testing.assert_array_equal(np.asarray(cache.keys.astype(mx.float32)), np.full((1, 1, 2, 4), 3))
