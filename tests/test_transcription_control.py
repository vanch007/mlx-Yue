"""Cancellation and resource failures must stop work before publishing success."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lyra.transcription import pipeline


def test_pre_cancel_does_not_start_decoder_or_load_weights(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    source.touch()
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(InterruptedError):
        pipeline.transcribe(source, tmp_path / "out", cancelled=lambda: True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("error", [InterruptedError, MemoryError])
def test_running_decoder_is_reaped_on_interrupt(monkeypatch, error):
    original, children, checks = subprocess.Popen, [], []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    def cancelled():
        checks.append(True)
        if len(checks) >= 3:
            if error is MemoryError:
                raise MemoryError("sampled pressure")
            return True
        return False

    monkeypatch.setattr(pipeline.subprocess, "Popen", spawn)
    with pytest.raises(error):
        pipeline.decode_audio([sys.executable, "-c", "import time; time.sleep(30)"], cancelled)
    assert len(children) == 1 and children[0].poll() is not None


def test_decoder_reports_failure_and_timeout():
    with pytest.raises(subprocess.CalledProcessError) as caught:
        pipeline.decode_audio([sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(7)"])
    assert caught.value.returncode == 7 and caught.value.stderr == b"bad"
    with pytest.raises(subprocess.TimeoutExpired):
        pipeline.decode_audio([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)


def test_decoder_drains_both_pipes_and_keeps_samples():
    command = [sys.executable, "-c",
               "import sys; sys.stderr.buffer.write(b'e'*200000); "
               "sys.stdout.buffer.write(b'\\x00\\x00\\x80\\x3f'*100000)"]
    np.testing.assert_array_equal(pipeline.decode_audio(command), np.ones(100000, np.float32))


def test_resource_failure_during_inference_prevents_success_publication(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.touch()
    failed = False

    def cancelled():
        if failed:
            raise MemoryError("pressure during inference")
        return False

    def infer(*args):
        nonlocal failed
        failed = True
        return np.array([1, 2]), False

    tokenizer = SimpleNamespace(normalize_prompts=lambda prompts: prompts)
    model = SimpleNamespace(config={"input_audio_length": 300, "time_hz": 10,
                                     "tokenizer_schema_version": "v1", "tokenizer_fingerprint": None})
    monkeypatch.setattr(pipeline, "decode_audio", lambda *a: np.zeros(1025, np.float32))
    monkeypatch.setattr(pipeline, "SheetSage2Tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(pipeline, "generate_tokens", infer)
    monkeypatch.setattr(pipeline, "decode_generated_tokens", lambda *a: ({"events": []}, None))
    monkeypatch.setattr(pipeline, "stitched_window_events", lambda *a: [])
    monkeypatch.setattr(pipeline, "export_result", lambda *a, **k: pytest.fail("exported after failure"))
    with pytest.raises(MemoryError, match="during inference"):
        pipeline.transcribe(source, tmp_path / "out", model=model, cancelled=cancelled)
    assert not (tmp_path / "out/result.json").exists()
    assert not (tmp_path / "out/windows.json").exists()


@pytest.mark.parametrize("entry", ["helper", "cover"])
def test_transcription_entry_propagates_guard_failure(tmp_path, monkeypatch, entry):
    from lyra.cli import parser
    from lyra.commands import cover
    from lyra.music_tools.transcribe import run

    class Guard:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def check(self):
            raise MemoryError("sampled pressure")

    def transcribe(audio, output, **kwargs):
        kwargs["cancelled"]()
        pytest.fail("continued after resource failure")

    monkeypatch.setattr("lyra.measure.GPUExecution", Guard)
    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    if entry == "helper":
        args = SimpleNamespace(audio="unused", output=tmp_path / "out", model="unused",
                               base_model="unused", offline=True, cache_dir=None, task="full",
                               preset="default", max_seconds=None, memory_budget_gib=24, quiet=True)
        call = run
    else:
        from contextlib import nullcontext
        monkeypatch.setattr("lyra.cli._resource_monitor", lambda args: nullcontext())
        monkeypatch.setattr("lyra.cli._pipeline", lambda *a: pytest.fail("rendered cover"))
        args = parser().parse_args(["cover", "--audio", "unused", "--style", "pop",
                                   "--lyrics", "hello", "--output", str(tmp_path / "out")])
        call = cover
    with pytest.raises(MemoryError, match="sampled pressure"):
        call(args)
    assert not (tmp_path / "out").exists()
