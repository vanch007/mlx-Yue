import numpy as np
import pytest

from lyra.nar import song_chunks
from lyra.pipeline import initial_noise


def test_noise_uses_full_seed_and_preserves_global_rng():
    np.random.seed(91)
    before = np.random.get_state()
    first = initial_noise(16, 2**40 + 7)
    np.testing.assert_array_equal(first, initial_noise(16, 2**40 + 7))
    assert not np.array_equal(first, initial_noise(16, 7))
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    with pytest.raises(ValueError):
        initial_noise(0, 7)


def test_chunks_share_exact_full_song_noise():
    chunks = song_chunks([151643], list(range(12)), 42, context=16)
    np.testing.assert_array_equal(np.concatenate([c.noise for c in chunks]), initial_noise(12, 42))
    assert [len(c.noise) for c in chunks] == [6, 6]
