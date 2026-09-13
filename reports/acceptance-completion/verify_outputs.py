"""Structural/audio checks, separate from subjective musical quality."""
import json
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from lyra.artifacts import load_artifacts
from lyra.music_tools.abc_tools import parse_abc
from yue2.storage import sha256_file, verify_result

ROOT = Path("outputs/acceptance-completion")
rows = []
for result in sorted(ROOT.rglob("result.json")):
    folder = result.parent
    row = {"path": str(folder)}
    try:
        data = json.loads(result.read_text())
        if (folder / "audio.flac").exists():
            verify_result(folder)
            if (folder / "noise.npy").exists():
                load_artifacts(folder)
            with sf.SoundFile(folder / "audio.flac") as audio:
                peak = square = count = near_clip = 0
                quiet_seconds = []
                for i, block in enumerate(audio.blocks(blocksize=48000, dtype="float32", always_2d=True)):
                    assert np.isfinite(block).all(), "Nonfinite samples"
                    peak = max(peak, float(np.abs(block).max()))
                    energy = float(np.square(block.astype(float)).sum())
                    square += energy
                    count += block.size
                    near_clip += int((np.abs(block) >= 0.9999).sum())
                    if np.sqrt(energy / block.size) < 1e-4:
                        quiet_seconds.append(i)
                assert count > 0 and square > 0, "Empty or all-zero audio"
                row.update(kind="song", seconds=audio.frames / audio.samplerate,
                           peak=peak, rms=(square / count)**0.5, near_full_scale_fraction=near_clip / count,
                           quiet_second_indices=quiet_seconds, truncated=data["truncated"],
                           naturally_ended=not any(data["truncated"].values()),
                           audio_sha256=sha256_file(folder / "audio.flac"))
        elif (folder / "windows.json").exists():
            windows = json.loads((folder / "windows.json").read_text())
            events = json.loads((folder / "events.json").read_text())["events"]
            times = [e["time"] for e in events]
            assert times == sorted(times) and all(0 <= t < data["duration_seconds"] for t in times)
            assert windows[0]["accept_start"] == 0
            assert abs(windows[-1]["accept_end"] - data["duration_seconds"]) < 1e-5
            assert all(a["accept_end"] == b["accept_start"] for a, b in zip(windows, windows[1:]))
            assert data["status"] == "complete" and not data["truncated"]
            for midi in folder.glob("*.mid"):
                mido.MidiFile(midi)
            row.update(kind="transcription", task=data["task"], seconds=data["duration_seconds"],
                       windows=len(windows), events=len(events), notes=data["melody_notes"],
                       measures=data["abc_measures"], event_windows=sorted({e["window_index"] for e in events}),
                       warnings=data["warnings"], diagnostics=data["diagnostics"])
        else:
            continue
        if (folder / "score.abc").exists():
            parse_abc((folder / "score.abc").read_text())
        row["status"] = "pass"
    except Exception as error:
        row.update(status="fail", error=f"{type(error).__name__}: {error}")
    rows.append(row)
report = {"scope": "Integrity, geometry, finite audio, natural endings, transcription coverage/export; not listening approval",
          "outputs": rows, "passed": sum(r["status"] == "pass" for r in rows),
          "failed": sum(r["status"] == "fail" for r in rows)}
Path("reports/acceptance-completion/output-checks.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: report[k] for k in ("passed", "failed")})
for row in rows:
    if row["status"] != "pass":
        print(row)
