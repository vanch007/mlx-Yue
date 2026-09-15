"""Regression coverage for the macOS 15 community report and fresh installs."""
import json
from types import SimpleNamespace

import pytest

from lyra import runtime


@pytest.mark.parametrize("macos,device", [
    ("14.2", "Apple M1"), ("15.7.7", "Apple M3 Ultra"),
    ("15.0", "Apple M4"), ("26.2", "Apple M5"), ("27.0", "Apple M3 Max"),
])
def test_supported_apple_silicon_versions(macos, device):
    assert runtime.platform_error("Darwin", "arm64", macos, device) is None


@pytest.mark.parametrize("system,machine,macos,device,message", [
    ("Linux", "aarch64", "", "", "Apple Silicon"),
    ("Darwin", "x86_64", "15.7.7", "", "Rosetta"),
    ("Darwin", "arm64", "14.1", "Apple M1", "14.2"),
    ("Darwin", "arm64", "26.1", "Apple M5 Max", "26.2"),
    ("Darwin", "arm64", "", "", "determine"),
])
def test_unsupported_versions_have_actionable_errors(system, machine, macos, device, message):
    assert message in runtime.platform_error(system, machine, macos, device)


def test_pipeline_on_macos15_reaches_model_validation(monkeypatch, tmp_path):
    import mlx.core as mx
    from lyra import pipeline

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: ("15.7.7", (), ""))
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"})

    def reached_model(*args):
        raise FileNotFoundError("reached-model-validation")

    monkeypatch.setattr(pipeline, "verify_conversion", reached_model)
    with pytest.raises(FileNotFoundError, match="reached-model-validation"):
        pipeline.YuE2Pipeline(tmp_path, tmp_path, progress=False)


def test_doctor_and_pipeline_use_the_same_runtime_rejection(monkeypatch, capsys):
    from lyra.commands import doctor

    monkeypatch.setattr(runtime, "runtime_status", lambda: {
        "system": "Darwin", "supported": False, "metal": False,
        "error": "MLX Metal is required",
    })
    assert doctor(SimpleNamespace(model=None, vae=None, output=None)) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["checks"]["supported_runtime"]
    with pytest.raises(RuntimeError, match="MLX Metal"):
        runtime.require_supported_runtime()
