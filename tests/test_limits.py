"""A checkpoint identity alone cannot bind a numerical calibration to a request."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_calibration_rejects_different_teacher_forced_tokens(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from limits import derive

    reference, anchor = tmp_path / "reference", tmp_path / "anchor"
    reference.mkdir()
    anchor.mkdir()
    model = {"checkpoint": "same-checkpoint"}
    (reference / "ar.json").write_text(json.dumps({"model": model}))
    (anchor / "invocation.json").write_text(json.dumps({"weights": model}))
    ids = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    np.savez(reference / "ar.npz", ids_5=ids, logits_5=np.zeros((1, 2), dtype=np.float32))
    np.savez(anchor / "ar.npz", ids_5=ids, logits_5=np.ones((1, 2), dtype=np.float32))
    calibrated = derive("ar", reference, anchor)
    assert calibrated["ar"]["tensors"]["logits_5"] == {"rms_error": 2, "max_abs": 2}

    changed = ids.copy()
    changed[-1] += 1
    np.savez(anchor / "ar.npz", ids_5=changed, logits_5=np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        derive("ar", reference, anchor)
