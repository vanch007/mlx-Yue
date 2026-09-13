"""Keep original FP32 attention on CPU to bound MPS reference workspaces."""
import subprocess
import time
import run_numerics as jobs

jobs.JOB_LOG = jobs.REPORT / "layerwise-cpu-anchor-jobs.json"
for name, extra in [
    ("ar-long-layerwise", []),
    ("ar-valid-prefix-layerwise", ["--source", str(jobs.OUT / "fixtures/long-plan")]),
]:
    output = jobs.OUT / f"{name}-fp32-cpu-attention"
    command = [".oracle/bin/python", "reports/acceptance-completion/layerwise_ar.py", "--model", jobs.MODEL,
               "--output", str(output), "--precision", "fp32", "--attention-device", "cpu", *extra]
    start = time.time()
    with (jobs.REPORT / f"{name}-fp32-cpu-attention.log").open("w") as log:
        result = subprocess.run(command, env=jobs.ENV, stdout=log, stderr=subprocess.STDOUT)
    print(name, "CPU FP32 attention", "exit", result.returncode, round(time.time() - start, 1), flush=True)
    if result.returncode == 0:
        jobs.compare(name, "ar", jobs.OUT / f"{name}-bf16", output, ["--prefill-chunk-size", "256"])
