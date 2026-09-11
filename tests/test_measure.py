"""Memory safety must survive shrinking old swap and interrupted workloads."""
from importlib.machinery import ModuleSpec
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from lyra import measure


def test_late_environment_change_cannot_hide_cached_tf32():
    # The flag alone is insufficient: another MLX user may have initialized
    # NAX before Lyra was imported. A fresh precise process is the control.
    script = """
import json, os
import mlx.core as mx
mx.set_default_device(mx.gpu)
operand = mx.full((128, 128), 1 + 2**-14, dtype=mx.float32)
loss = mx.max(mx.abs(operand @ mx.eye(128) - operand)).item()
os.environ["MLX_ENABLE_TF32"] = "0"
from lyra.measure import GPUExecution
try:
    with GPUExecution():
        accepted = True
except RuntimeError:
    accepted = False
print(json.dumps({"precision_loss": loss, "accepted": accepted}))
"""
    results = []
    for primed in ("0", "1"):
        process = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "MLX_ENABLE_TF32": primed},
            text=True, capture_output=True, check=True,
        )
        results.append(json.loads(process.stdout))
    assert results[0]["precision_loss"] == 0
    assert results[0]["accepted"]
    if results[1]["precision_loss"] == 0:
        pytest.skip("This GPU does not expose the M5 reduced-precision path")
    assert not results[1]["accepted"]


@pytest.fixture
def quiet_machine(monkeypatch):
    sample = {
        "physical_footprint_bytes": 256 * 2**20,
        "system_memory_pressure_level": 1,
        "system_available_bytes": 12 * 2**30,
        "system_swap_used_bytes": 5 * 2**30,
        "system_swap_in_bytes": 100 * 2**30,
        "system_swap_out_bytes": 120 * 2**30,
    }
    monkeypatch.setattr(measure, "memory_snapshot", lambda: dict(sample))
    monkeypatch.setattr(measure, "power_source", lambda: {"ac_connected": True})
    # These tests exercise process policy, not framework allocator APIs.
    monkeypatch.setattr(measure.GPUExecution, "_apply_limits", lambda self: None)
    return sample


def test_old_swap_is_allowed_but_new_swapouts_abort_even_when_used_swap_shrinks(quiet_machine):
    with measure.GPUExecution() as guard:
        guard.check()  # Existing five GiB of idle swap is not a failure.
    with pytest.raises(MemoryError):
        with measure.GPUExecution() as guard:
            quiet_machine["system_swap_used_bytes"] -= 2**30
            quiet_machine["system_swap_out_bytes"] += 2**30
            guard.monitor.thread.join(timeout=2)
            guard.check()
    # The failed context must release ownership so a subsequent job can start.
    with measure.GPUExecution() as next_job:
        next_job.check()


def test_resource_evidence_is_readable_before_exit_and_survives_interruption(tmp_path, quiet_machine):
    log, report = tmp_path / "resources.jsonl", tmp_path / "resources.json"
    with pytest.raises(InterruptedError):
        with measure.ResourceMonitor(log_path=log, report_path=report):
            first = json.loads(log.read_text().splitlines()[0])
            assert first["system_swap_used_bytes"] == 5 * 2**30
            raise InterruptedError("Stopped the workload")
    saved = json.loads(report.read_text())
    assert saved["exception"].split(":", 1)[0] == "InterruptedError"
    assert saved["samples"] == [json.loads(line) for line in log.read_text().splitlines()]


def test_existing_resource_evidence_is_never_mixed_with_a_new_run(tmp_path, quiet_machine):
    log, report = tmp_path / "resources.jsonl", tmp_path / "resources.json"
    original = '{"previous_run": true}\n'
    log.write_text(original)
    with pytest.raises(FileExistsError):
        with measure.ResourceMonitor(log_path=log, report_path=report):
            pytest.fail("An existing evidence path must reject the new run")
    assert log.read_text() == original
    assert not report.exists()


def test_monitor_preserves_process_evidence_while_backends_are_importing(tmp_path, monkeypatch):
    # Import machinery publishes sys.modules entries before their APIs are ready.
    for name in ("mlx.core", "torch"):
        partial = ModuleType(name)
        partial.__spec__ = ModuleSpec(name, loader=None)
        partial.__spec__._initializing = True
        monkeypatch.setitem(sys.modules, name, partial)
    report = tmp_path / "import.resources.json"
    with measure.ResourceMonitor(report_path=report):
        snapshot = measure.memory_snapshot()
        assert "physical_footprint_bytes" in snapshot
        # Unavailable framework counters must be omitted, not reported as zero.
        assert "mlx_active_bytes" not in snapshot
        assert "mps_active_bytes" not in snapshot
    saved = json.loads(report.read_text())
    assert saved["monitor_error"] is None
    assert saved["exception"] is None
