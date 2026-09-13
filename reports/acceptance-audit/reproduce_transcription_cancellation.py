"""CPU-only reproductions for transcription cancellation/resource polling gaps."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile


def pre_cancel_still_decodes_audio():
    from lyra.transcription import pipeline

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=(b"\0\0\0\0" * 1025))

    class FakeModel:
        config = {
            "input_audio_length": 300.0,
            "sampling_rate": 24000,
            "time_hz": 100,
            "tokenizer_schema_version": "v1",
            "tokenizer_fingerprint": "unused",
        }

    with tempfile.TemporaryDirectory() as root:
        source = Path(root) / "input.wav"
        source.write_bytes(b"placeholder")
        with patch.object(pipeline.subprocess, "run", fake_run):
            try:
                pipeline.transcribe(
                    source,
                    Path(root) / "out",
                    model=FakeModel(),
                    cancelled=lambda: True,
                )
            except Exception:
                pass
    assert calls, "pre-cancelled transcription did not launch ffmpeg"
    print("PASS pre-cancelled transcription launched ffmpeg before checking cancellation")


def helper_writes_before_resource_failure():
    from lyra.music_tools import transcribe as helper
    import lyra.measure
    import lyra.transcription.pipeline

    state = {"checks": 0, "artifact_written": False}

    class FakeGuard:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def check(self):
            state["checks"] += 1
            raise MemoryError("latched sampled resource failure")

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise MemoryError("latched sampled resource failure")

    def fake_transcribe(audio, output, **kwargs):
        output = Path(output)
        output.mkdir()
        (output / "result.json").write_text("{}")
        state["artifact_written"] = True
        return dict(status="complete", backend="mlx", duration_seconds=1.0,
                    truncated=False, elapsed_seconds=0.1)

    args = SimpleNamespace(
        audio=Path("unused.wav"), output=None, model="model", base_model="base",
        offline=True, cache_dir=None, task="melody-full", preset="default",
        max_seconds=None, memory_budget_gib=24, quiet=True,
    )
    with tempfile.TemporaryDirectory() as root:
        args.output = Path(root) / "out"
        with patch.object(lyra.measure, "GPUExecution", FakeGuard), patch.object(
            lyra.transcription.pipeline, "transcribe", fake_transcribe
        ):
            try:
                helper.run(args)
            except MemoryError:
                pass
    assert state["artifact_written"]
    assert state["checks"] == 0
    print("PASS helper wrote result before guard exit; GPUExecution.check was never called")


if __name__ == "__main__":
    pre_cancel_still_decodes_audio()
    helper_writes_before_resource_failure()
