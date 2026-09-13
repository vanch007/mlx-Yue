"""Serial, immutable numerical acceptance captures for the local M3 runtime."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path("outputs/acceptance-completion")
REPORT = Path("reports/acceptance-completion")
MODEL = "models/hf-cache/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19"
VAE = json.loads(Path("models/paths.json").read_text())["vae"]
LEGACY = json.loads(Path("outputs/oracle-legacy/invocation.json").read_text())["args"]["vae"]
ENV = {**os.environ, "PYTHONPATH": "src:vendor/yue/src", "MLX_ENABLE_TF32": "0"}
records = []
JOB_LOG = REPORT / "numerical-jobs.json"


def run(name, env, tool, *args):
    command = [f".{env}/bin/python", f"tools/{tool}.py", *map(str, args)]
    row = dict(name=name, command=command, status="running", start=time.time())
    records.append(row)
    JOB_LOG.write_text(json.dumps(records, indent=2) + "\n")
    with (REPORT / f"{name}.log").open("w") as log:
        result = subprocess.run(command, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    row.update(status="pass" if result.returncode == 0 else "fail", exit_code=result.returncode,
               seconds=time.time() - row["start"])
    JOB_LOG.write_text(json.dumps(records, indent=2) + "\n")
    print(name, row["status"], round(row["seconds"], 1), flush=True)
    return result.returncode == 0


def compare(name, stage, oracle, calibration, extra=()):
    limits = REPORT / f"{name}-limits.json"
    if not run(name + "-limits", "runtime", "limits", stage, "--oracle", oracle,
               "--calibration", calibration, "--output", limits):
        return
    actual = OUT / (name + "-mlx")
    run(name + "-fidelity", "runtime", "fidelity", stage, "--oracle", oracle,
        "--model", "models/converted", "--limits", limits, "--output", actual,
        "--require-ac", *extra)
    if (actual / f"{stage}-actual.npz").exists():
        run(name + "-anchor", "runtime", "limits", stage, "--oracle", oracle,
            "--calibration", calibration, "--actual", actual / f"{stage}-actual.npz",
            "--limits", limits, "--output", REPORT / f"{name}-anchor.json")


def main():
    compare("ar-short", "ar", "outputs/oracle-ar-local", OUT / "calibration-ar-short")
    for name, lengths, extra in [
        ("ar-negative", [128, 1024], ["--branch", "negative"]),
        ("ar-long", [6144, 12000], []),
        ("ar-song-negative", [128, 1024, 6144], ["--branch", "negative", "--source", "outputs/precision-comparison/full-song-bf16"]),
        ("ar-long-prefix", [12000], ["--source", str(OUT / "fixtures/long-plan")]),
    ]:
        oracle, calibration = OUT / (name + "-torch"), OUT / (name + "-fp32")
        if not run(name + "-reference", "oracle", "oracle", "ar", "--model", MODEL,
                   "--vae", VAE, "--lengths", *lengths, *extra, "--output", oracle, "--require-ac"):
            continue
        if not run(name + "-calibration", "oracle", "calibrate", "ar", "--model", MODEL,
                   "--oracle", oracle, "--output", calibration, "--require-ac"):
            continue
        compare(name, "ar", oracle, calibration)

    oracle, calibration = Path("outputs/oracle-nar-local"), OUT / "nar-fp32"
    if run("nar-calibration", "oracle", "calibrate", "nar", "--model", MODEL,
           "--oracle", oracle, "--output", calibration, "--require-ac"):
        compare("nar", "nar", oracle, calibration)

    for name, vae in [("vae-default-512", VAE), ("vae-legacy-512", LEGACY)]:
        oracle = OUT / (name + "-torch")
        latents = OUT / "fixtures/vae-512.npy"
        if run(name + "-reference", "oracle", "oracle", "vae", "--model", MODEL,
               "--vae", vae, "--latents", latents, "--output", oracle, "--require-ac"):
            compare(name, "vae", oracle, oracle, ["--vae", vae, "--latents", str(latents)])


if __name__ == "__main__":
    main()
