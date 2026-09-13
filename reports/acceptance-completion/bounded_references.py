"""Final references with per-layer workspace reclamation, original 16 GiB guard."""
import subprocess
import run_numerics as jobs

jobs.JOB_LOG = jobs.REPORT / "bounded-reference-jobs.json"
for name, extra in [
    ("ar-long-layer-cleanup", []),
    ("ar-valid-prefix-layer-cleanup", ["--source", str(jobs.OUT / "fixtures/long-plan")]),
]:
    oracle, anchor = jobs.OUT / (name + "-torch"), jobs.OUT / (name + "-fp32")
    if not jobs.run(name + "-reference", "oracle", "oracle", "ar", "--model", jobs.MODEL,
                    "--vae", jobs.VAE, "--lengths", 12000, *extra, "--prefill-chunk-size", 256,
                    "--output", oracle, "--require-ac"):
        continue
    if jobs.run(name + "-calibration", "oracle", "calibrate", "ar", "--model", jobs.MODEL,
                "--oracle", oracle, "--output", anchor, "--require-ac"):
        jobs.compare(name, "ar", oracle, anchor, ["--prefill-chunk-size", "256"])

for name, command, cwd in [
    ("installed-wheel-retry", [str(jobs.ROOT / "outputs/acceptance-audit-env/bin/python"), "-I",
       str(jobs.REPORT.resolve() / "test_installed_pipeline.py")], "/tmp"),
    ("reference-full-song-bounded", [".oracle/bin/python", "tools/reference_song.py", "--request",
       "outputs/precision-comparison/full-song-bf16/request.json", "--model", jobs.MODEL,
       "--vae", jobs.VAE, "--output", str(jobs.OUT / "reference-full-song-bounded")], jobs.ROOT),
]:
    with (jobs.REPORT / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=cwd, env=jobs.ENV, stdout=log, stderr=subprocess.STDOUT)
    print(name, "exit", result.returncode, flush=True)
