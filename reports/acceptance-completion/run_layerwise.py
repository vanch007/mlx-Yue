"""Matched full-prefix BF16/FP32 references and normal cached MLX comparisons."""
import subprocess
import time
import run_numerics as jobs

jobs.JOB_LOG = jobs.REPORT / "layerwise-comparison-jobs.json"
for name, extra in [
    ("ar-long-layerwise", []),
    ("ar-valid-prefix-layerwise", ["--source", str(jobs.OUT / "fixtures/long-plan")]),
]:
    directories = []
    for precision in ("bf16", "fp32"):
        output = jobs.OUT / f"{name}-{precision}"
        command = [".oracle/bin/python", "reports/acceptance-completion/layerwise_ar.py",
                   "--model", jobs.MODEL, "--output", str(output), "--precision", precision, *extra]
        start = time.time()
        with (jobs.REPORT / f"{name}-{precision}.log").open("w") as log:
            result = subprocess.run(command, env=jobs.ENV, stdout=log, stderr=subprocess.STDOUT)
        print(name, precision, "exit", result.returncode, round(time.time() - start, 1), flush=True)
        if result.returncode:
            break
        directories.append(output)
    if len(directories) == 2:
        jobs.compare(name, "ar", directories[0], directories[1], ["--prefill-chunk-size", "256"])
