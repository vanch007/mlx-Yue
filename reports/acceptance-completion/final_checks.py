"""Final bounded long-context capture and acceleration diagnostic jobs."""
import os
from pathlib import Path
import subprocess
import run_numerics as jobs

jobs.JOB_LOG = jobs.REPORT / "final-numerical-jobs.json"
run, OUT, REPORT = jobs.run, jobs.OUT, jobs.REPORT
for name, extra in [
    ("ar-long-prefill256", []),
    ("ar-valid-prefix-prefill256", ["--source", str(OUT / "fixtures/long-plan")]),
]:
    oracle, calibration = OUT / (name + "-torch"), OUT / (name + "-fp32")
    if not run(name + "-reference", "oracle", "oracle", "ar", "--model", jobs.MODEL,
               "--vae", jobs.VAE, "--lengths", 12000, *extra, "--prefill-chunk-size", 256,
               "--output", oracle, "--require-ac"):
        continue
    if not run(name + "-calibration", "oracle", "calibrate", "ar", "--model", jobs.MODEL,
               "--oracle", oracle, "--output", calibration, "--require-ac"):
        continue
    jobs.compare(name, "ar", oracle, calibration, ["--prefill-chunk-size", "256"])

for name, stage, oracle, limits in [
    ("promoted-negative", "ar", OUT / "ar-negative-torch", REPORT / "ar-negative-limits.json"),
    ("promoted-song-negative", "ar", OUT / "ar-song-negative-bounded-torch", REPORT / "ar-song-negative-bounded-limits.json"),
    ("promoted-nar", "nar", Path("outputs/oracle-nar-local"), REPORT / "nar-limits.json"),
]:
    command = [".runtime/bin/python", "reports/acceptance-completion/diagnose_attention.py", stage,
               "--model", "models/converted", "--oracle", str(oracle), "--limits", str(limits),
               "--output", str(OUT / name), "--require-ac"]
    with (REPORT / f"{name}.log").open("w") as log:
        result = subprocess.run(command, env=jobs.ENV, stdout=log, stderr=subprocess.STDOUT)
    print(name, "exit", result.returncode, flush=True)

with (REPORT / "installed-wheel.log").open("w") as log:
    result = subprocess.run([str(jobs.ROOT / "outputs/acceptance-audit-env/bin/python"), "-I",
        str(jobs.ROOT / "reports/acceptance-completion/test_installed_pipeline.py")],
        cwd="/tmp", env={**os.environ, "MLX_ENABLE_TF32": "0"}, stdout=log, stderr=subprocess.STDOUT)
print("installed-wheel", "exit", result.returncode, flush=True)
