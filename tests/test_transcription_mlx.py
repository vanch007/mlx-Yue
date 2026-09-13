"""Independent FP32 Torch oracle fixtures; no Torch needed to run these tests."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from lyra.bart import BartDecoder
from lyra.mert import MERT2, apply_layers, convert_mert_weights

FIXTURES = Path(__file__).parent / "fixtures"


def test_mert_matches_official_frontend_subsampling_and_conformer():
    folder = FIXTURES / "mert"
    model = MERT2(json.loads((folder / "config.json").read_text()))
    with np.load(folder / "weights.npz") as weights:
        model.load_weights(convert_mert_weights(dict(weights)), strict=True)
    with np.load(folder / "values.npz") as data:
        waveform = mx.array(data["x"])
        mel = model.feature_extractor(waveform)
        sub = apply_layers(model.subsampling_module, mel)
        for actual, key in ((mel, "mel"), (sub, "sub"), (model(waveform), "out")):
            np.testing.assert_allclose(np.array(actual), data[key], atol=3e-5, rtol=3e-4)


def test_bart_matches_reference_and_cached_decode():
    folder = FIXTURES / "bart"
    model = BartDecoder(dict(hidden_size=16, num_attention_heads=2,
                             intermediate_size=32, decoder_layers=2, max_output_seq_len=64))
    with np.load(folder / "weights.npz") as weights:
        model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
    with np.load(folder / "values.npz") as data:
        x, memory = mx.array(data["x"]), mx.array(data["memory"])
        actual, _ = model(x, memory)
        np.testing.assert_allclose(np.array(actual), data["y"], atol=2e-5, rtol=1e-4)
        _, cache = model(x[:, :-1], memory)
        last, _ = model(x[:, -1:], memory, cache)
        np.testing.assert_allclose(np.array(last), data["y"][:, -1:], atol=2e-5, rtol=1e-4)


def test_transcription_window_coverage():
    from lyra.transcription.upstream.windows import sliding_window_plan
    windows = sliding_window_plan(610, 300, 200, 100)
    assert windows[0]["accept_start"] == 0
    assert windows[-1]["accept_end"] >= 610
    for before, after in zip(windows, windows[1:]):
        assert before["accept_end"] == after["accept_start"]
