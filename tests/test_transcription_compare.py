import json
from pathlib import Path

import numpy as np
import pytest

from yue2.storage import sha256_file


def test_trajectory_comparison_detects_divergence_and_wrong_inputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from compare_transcription import compare

    ref, port = tmp_path / "torch", tmp_path / "mlx"

    def capture(path, tokens, input_hash="same"):
        path.mkdir(exist_ok=True)
        np.savez(path / "values.npz", generated=np.array(tokens, np.int64),
                 memory=np.ones(3, np.float32), logits=np.ones(3, np.float32), cached=np.ones(3, np.float32))
        (path / "report.json").write_text(json.dumps({
            "values_sha256": sha256_file(path / "values.npz"), "backend": path.name,
            "input_sha256": input_hash, "weights_sha256": "same", "task": "full", "complete_generation": True,
        }))

    capture(ref, [1, 2, 3])
    capture(port, [1, 2, 3])
    assert compare(ref, port)["status"] == "pass"
    capture(port, [1, 4, 3])
    result = compare(ref, port)
    assert result["status"] == "fail" and result["first_mismatch"] == 1
    capture(port, [1, 2, 3], input_hash="other")
    with pytest.raises(ValueError, match="input_sha256"):
        compare(ref, port)
