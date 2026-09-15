# Historical numerical reference

The pinned `requirements.in` / `requirements.txt` reproduce the earlier Torch
2.11.0 / Transformers 4.57.6 numerical experiments. They are archived evidence,
not current installation guidance. These old pins are **not covered** by the
clean current-runtime audit; they include versions with known advisories.

Use the root lockfile and `uv sync --frozen --no-dev` for music generation, or
`uv sync --frozen --extra transcription` for the current development tests.
The latter uses Torch 2.13.0 / Transformers 5.10.4. Reusing old numerical baselines
requires preserving their exact environment identity, so the historical pins
have not been silently rewritten.

See [the dependency/platform verification report](../reports/x-feedback-20260915/REPORT.md).
