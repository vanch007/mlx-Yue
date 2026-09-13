# Transcription cancellation/resource reproduction

Commit reviewed: `8aa09f5f08bbdfd8d9a391afff0bd28d280df3fd`

Run (CPU-only):

```console
uv run python reports/acceptance-audit/reproduce_transcription_cancellation.py
PASS pre-cancelled transcription launched ffmpeg before checking cancellation
PASS helper wrote result before guard exit; GPUExecution.check was never called
```

The first reproduction replaces `subprocess.run` and supplies an already-true
`cancelled` callback. The fake FFmpeg call is still observed before cancellation.

The second replaces the GPU guard and transcription model work. The fake guard
models a sampled resource failure latched during work and raised on context exit.
The helper writes `result.json` before that exit and never calls `guard.check()`.
This establishes call ordering without loading MLX weights or doing GPU work.

Static evidence:

- `src/lyra/transcription/pipeline.py:71-81` runs FFmpeg and loads the model before
  the first reachable cancellation check.
- `src/lyra/music_tools/transcribe.py:11-15` does not retain the entered guard or
  pass a cancellation/resource callback.
- `src/lyra/commands.py:127-132` has the same issue in cover transcription.
- `src/lyra/measure.py:297-313,386-393,400-405` latches sampled failures; callers
  must poll `check()` for timely interruption, otherwise exit reports them later.

No model checkpoint or GPU inference is used by this reproduction.
