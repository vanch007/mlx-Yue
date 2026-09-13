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


def test_vae_fp64_bounds_use_reference_roundoff_and_bind_inputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from limits import derive

    reference, anchor = tmp_path / "reference", tmp_path / "anchor"
    reference.mkdir()
    anchor.mkdir()
    np.savez(reference / "vae.npz", full=np.ones(4, np.float32), core_64=np.ones(4, np.float32))
    np.savez(anchor / "vae.npz", full=np.ones(4, np.float64) + 1e-7)
    (reference / "vae.json").write_text(json.dumps({"weights": {"hash": "same"}}))
    (reference / "invocation.json").write_text(json.dumps({"input_sha256": {"latents": "input"}}))
    metadata = {"precision": "float64", "weights": {"hash": "same"}, "latents_sha256": "input"}
    (anchor / "invocation.json").write_text(json.dumps(metadata))
    limits = derive("vae", reference, anchor, vae_fp64=True)
    assert limits["vae"]["audio"]["max_abs"] == pytest.approx(2e-7)
    assert not limits["calibration"]["uses_port_outputs"]
    metadata["latents_sha256"] = "different"
    (anchor / "invocation.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="match"):
        derive("vae", reference, anchor, vae_fp64=True)
