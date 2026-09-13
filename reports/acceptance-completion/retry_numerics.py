"""Correct reference capture geometry; retain every initial failure unchanged."""
import json
from pathlib import Path
import run_numerics as jobs

jobs.JOB_LOG = jobs.REPORT / "numerical-retry-jobs.json"
run, OUT, REPORT = jobs.run, jobs.OUT, jobs.REPORT
for name, lengths, extra in [
    ("ar-long-bounded", [6144, 12000], []),
    ("ar-song-negative-bounded", [128, 1024, 6144], ["--branch", "negative", "--source", "outputs/precision-comparison/full-song-8bit"]),
    ("ar-long-prefix-bounded", [12000], ["--source", str(OUT / "fixtures/long-plan")]),
]:
    oracle, calibration = OUT / (name + "-torch"), OUT / (name + "-fp32")
    if not run(name + "-reference", "oracle", "oracle", "ar", "--model", jobs.MODEL,
               "--vae", jobs.VAE, "--lengths", *lengths, *extra, "--prefill-chunk-size", 1024,
               "--output", oracle, "--require-ac"):
        continue
    if not run(name + "-calibration", "oracle", "calibrate", "ar", "--model", jobs.MODEL,
               "--oracle", oracle, "--output", calibration, "--require-ac"):
        continue
    jobs.compare(name, "ar", oracle, calibration, ["--prefill-chunk-size", "1024"])

for name, vae in [("vae-default-512", jobs.VAE), ("vae-legacy-512", jobs.LEGACY)]:
    oracle, anchor = OUT / (name + "-torch"), OUT / (name + "-fp64")
    latents, limits = OUT / "fixtures/vae-512.npy", REPORT / (name + "-fp64-limits.json")
    if not run(name + "-fp64-calibration", "oracle", "calibrate_vae", "--vae", vae,
               "--latents", latents, "--output", anchor):
        continue
    if not run(name + "-fp64-limits", "runtime", "limits", "vae", "--oracle", oracle,
               "--calibration", anchor, "--vae-fp64", "--output", limits):
        continue
    actual = OUT / (name + "-fp64-budget-mlx")
    run(name + "-fp64-fidelity", "runtime", "fidelity", "vae", "--oracle", oracle,
        "--vae", vae, "--latents", latents, "--limits", limits, "--output", actual, "--require-ac")
    if (actual / "vae-actual.npz").exists():
        run(name + "-fp64-anchor", "runtime", "limits", "vae", "--oracle", oracle,
            "--calibration", anchor, "--actual", actual / "vae-actual.npz", "--limits", limits,
            "--output", REPORT / (name + "-fp64-anchor.json"))

paths = json.loads(Path("models/transcription-paths.json").read_text())
for backend, env in [("torch", "oracle"), ("mlx", "runtime")]:
    run("transcription-generated-" + backend, env, "transcription_oracle", backend,
        "--model", paths["m-a-p/SheetSage2"], "--parent", paths["m-a-p/MERT-v2-FullSong"],
        "--input", "outputs/transcription-oracle-input.npz", "--generate", "--task", "melody-full",
        "--output", OUT / ("transcription-generated-" + backend))

run("transcription-generated-compare", "runtime", "compare_transcription",
    "--reference", OUT / "transcription-generated-torch", "--actual", OUT / "transcription-generated-mlx",
    "--output", REPORT / "transcription-generated-comparison.json")
