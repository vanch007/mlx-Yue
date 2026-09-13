"""Run complete mode, transcription, cover and reference checks serially."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path("outputs/acceptance-completion")
REPORT = Path("reports/acceptance-completion")
paths = json.loads(Path("models/paths.json").read_text())
transcription = json.loads(Path("models/transcription-paths.json").read_text())
model, vae = paths["model"], paths["vae"]
tm, base = transcription["m-a-p/SheetSage2"], transcription["m-a-p/MERT-v2-FullSong"]
ENV = {**os.environ, "PYTHONPATH": "src:vendor/yue/src", "MLX_ENABLE_TF32": "0"}
records = []


def run(name, command):
    row = dict(name=name, command=list(map(str, command)), status="running", start=time.time())
    records.append(row)
    (REPORT / "workflow-jobs.json").write_text(json.dumps(records, indent=2) + "\n")
    with (REPORT / f"{name}.log").open("w") as log:
        result = subprocess.run(row["command"], env=ENV, stdout=log, stderr=subprocess.STDOUT)
    row.update(status="pass" if result.returncode == 0 else "fail", exit_code=result.returncode,
               seconds=time.time() - row["start"])
    (REPORT / "workflow-jobs.json").write_text(json.dumps(records, indent=2) + "\n")
    print(name, row["status"], round(row["seconds"], 1), flush=True)


for task in ("full", "melody-full", "melody-vocal"):
    run("transcribe-" + task, [".runtime/bin/python", "-m", "lyra.music_tools.transcribe",
        OUT / "fixtures/multiwindow.flac", "--output", OUT / ("transcribe-" + task),
        "--model", tm, "--base-model", base, "--offline", "--task", task])

run("complete-modes", [".runtime/bin/python", "tools/acceptance.py", "modes",
    "--complete-modes", "--model", model, "--vae", vae, "--require-ac",
    "--output", OUT / "complete-modes"])
run("complete-cover", [".runtime/bin/yue2-mlx", "cover", "--request", OUT / "fixtures/cover-request.json",
    "--audio", "outputs/precision-comparison/fixed-score-bf16/audio.flac", "--task", "full",
    "--transcription-model", tm, "--base-model", base, "--model", model,
    "--vae", vae, "--offline", "--require-ac", "--output", OUT / "complete-cover"])

for name in ("fixed-score-bf16", "full-song-bf16"):
    run("reference-" + name, [".oracle/bin/python", "tools/reference_song.py",
        "--request", Path("outputs/precision-comparison") / name / "request.json",
        "--model", "models/hf-cache/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19",
        "--vae", vae, "--output", OUT / ("reference-" + name)])

run("sustain-retry", [".runtime/bin/python", "tools/acceptance.py", "sustain",
    "--model", model, "--vae", vae, "--require-ac", "--output", OUT / "sustain-bf16-retry"])
