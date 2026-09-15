# Community feedback: dependencies and macOS compatibility

Date: 2026-09-15. Baseline: `1a7c4ca`.

[PIYO's report](https://x.com/C_of_Creativity/status/2099534049991921704)
identified known dependency advisories and compatibility changes needed on M3
Ultra / macOS 15.7.7. Their single-run timings for a 220-second song were 206
seconds at 32 steps and 130 seconds at 8 steps, including load/save. These are
**the reporter's measurements of a modified installation**, not measurements
reproduced here. The post explicitly did not allege finding malware in mlx-Yue.

## Fixes

- Replace Transformers 5.0.0 with 5.10.4. The former matches
  [CVE-2026-4372](https://github.com/advisories/GHSA-29pf-2h5f-8g72),
  [CVE-2026-5241](https://github.com/advisories/GHSA-fgcw-684q-jj6r) and
  [CVE-2026-9856](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w).
- Replace Pretty MIDI 0.2.10 with 0.2.11.post0, which uses `importlib_resources`
  instead of `pkg_resources`. This removes the runtime dependency on old
  Setuptools; simply upgrading Setuptools would break the old MIDI import.
- Update development-only Torch 2.10.0 to 2.13.0 and Setuptools 78.1.1 to 83.0.0.
  The old Setuptools version matches
  [CVE-2026-59890](https://github.com/advisories/GHSA-h35f-9h28-mq5c).
- Make generation-only installation the quickstart default; transcription/cover
  remain an explicit extra, and development/reference dependencies are separate.
- Regenerate `uv.lock` with the actual distribution name `mlx-yue`. Previously,
  `uv export --frozen` failed because the lock still contained `lyra-yue2`.
- Fix pipeline runtime metadata lookup to use `mlx-yue`. A clean wheel installation
  must not depend on stale `lyra-yue2` metadata from an older editable install.
- Share platform policy between `doctor` and generation. M1–M4 require macOS
  14.2+; only M5 retains the macOS 26.2+ requirement. Reject Intel/Rosetta and
  missing Metal explicitly. Preserve existing GPU precision safeguards.

[MLX's installation documentation](https://ml-explore.github.io/mlx/build/html/install.html)
and the 0.32.2 wheel metadata include older macOS targets; a blanket macOS 26.2
rejection was not appropriate for M3 Ultra. Unit tests simulate 15.7.7 and verify
that the real pipeline reaches model validation. **Physical M3 Ultra / macOS
15.7.7 retesting is still pending**; additional reporter-side patches have not
been supplied, so this is not a claim to have reproduced every modification.

## Dependency audit

Tool: `pip-audit==2.10.1`, PyPI advisory service, Python 3.12, macOS arm64.
Requirements are exported from the current lock without installing project
packages in the audit tool's environment. No findings are ignored; a skipped
package makes the helper return failure.

| Profile | Packages checked | Known advisory matches | Skipped |
|---|---:|---:|---:|
| Generation | 42 | 0 | 0 |
| Generation + transcription | 49 | 0 | 0 |
| Development + transcription | 58 | 0 | 0 |

The old universal lock scan checked 77 public packages and returned 9 raw
records in 3 packages, representing 6 distinct CVEs; the service returned some
records twice. This is a dependency-version scan, not proof that those code paths
were reachable in music generation, and not a malware scan or a security guarantee.

Historical `oracle/requirements.txt` remains an explicitly archived numerical
reference. Its separate audit returns findings in Accelerate, Setuptools, Torch
and Transformers. It is **not** included in the clean profiles above and is not
recommended for new installations. Changing those pins silently would invalidate
the provenance of earlier numerical measurements; use the current development
group for ongoing tests. See [oracle documentation](../../oracle/README.md).

Raw evidence: [before](audit-before.json), [runtime](runtime-audit.json),
[transcription](transcription-audit.json), [development](dev-audit.json),
[historical oracle](oracle-historical-audit.json).

Repeat the audit from the repository root:

```bash
python tools/audit_dependencies.py --profile runtime --output outputs/dependency-audit
python tools/audit_dependencies.py --profile transcription --output outputs/dependency-audit
python tools/audit_dependencies.py --profile dev --output outputs/dependency-audit
```

## Functional verification

The release candidate was assembled from the published baseline plus these
scoped fixes, excluding unrelated unfinished local RealAudio changes. Its wheel
was installed in a fresh Torch-free environment without the legacy package name.
All three entrypoints (`mlx-yue`, `lyra`, `yue2-mlx`) returned a passing `doctor`.
The normal local generation and development environments were also synchronized.

- **32-step regression:** identical quickstart input and seed, 8-bit AR, 15.9987
  seconds of audio. Before/after semantic tokens, acoustic latents and retained
  noise and the saved FLAC are bit-identical. This intentionally capped short input checks numerical
  regression, not full-song completion or subjective quality.
- **8-step full-song regression:** `examples/full-song.json`, seed 12300,
  generated full symbolic score, 8-bit AR, 8 acoustic steps; fresh installed wheel.
  204.2787 seconds of stereo 48 kHz FLAC; both planning and semantic stages ended
  naturally. Finite samples, RMS 0.15309. PyTorch neither installed nor imported.
- **Timing:** 235.8869 seconds including Python-side import/load/generation/save;
  RTF 1.15473. M3 Max 128 GiB, macOS 27.0, AC at start/end. Sampled peak process
  physical footprint 11.18 GiB; OS lifetime peak 11.34 GiB; MLX peak 10.48 GiB.
  One functional run with other host activity, not a controlled speed comparison.
- **Optional MIDI:** note write/read roundtrip passes in the updated Torch-free
  transcription environment, without Setuptools installed.
- **Regression suite:** 119 passed, 1 skipped, 28 subtests passed; Ruff passed.
  The M5-specific reduced-precision environment-switch test was skipped because
  this M3 GPU does not expose that execution path; other precision tests passed.

The first run with Torch 2.13 exposed two changes in its MPS reference behavior:
the cancellation-sensitive attention output and comparison of a Python `.749`
threshold against BF16 `.75`. We replaced those two unstable reference checks
with an independent FP64 attention formula (rounding only the final output) and
an exact-quarter-probability CPU FP32 sampling oracle. The assertions still
require bit equality and exactly three retained tokens, respectively. A deliberate
BF16 probability-rounding calculation is also checked to differ from the expected
attention output. Production MLX arithmetic was not changed to match Torch's new
rounding. The full updated suite then passed.

Evidence: [short regression](short-generation.json), [full generation](full-generation.json),
[resources](full-resources-summary.json), [doctor](doctor-after.json),
[entrypoints](entrypoints.json), [MIDI](midi-smoke.json), [tests](tests-summary.json).
Audio is retained locally at `outputs/x-feedback-20260915-full/audio.flac`;
its SHA-256 is included in the generation evidence. It is not added to Git.
