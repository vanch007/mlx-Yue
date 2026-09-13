"""Run 8-step fast mode synthesis for all 15 benchmark cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/official-fullsong"
OUT = ROOT / "outputs/official_fullsong"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_case_8step(case_id, pipe=None):
    from lyra.artifacts import load_artifacts
    from lyra.measure import ResourceMonitor
    from yue2.protocol import GenerationConfig

    case_dir = OUT / case_id
    if (case_dir / "attempt-2/song").exists():
        song_dir = case_dir / "attempt-2/song"
        parent_measurement = read(case_dir / "attempt-2/measurement.json")
    else:
        song_dir = case_dir / "song"
        parent_measurement = read(case_dir / "measurement.json")

    dest = case_dir / "fast_8step"
    dest.mkdir(parents=True, exist_ok=True)

    saved = load_artifacts(song_dir)
    print(f"\n--- [8-Step Fast Mode] Starting: {case_id} ({len(saved.semantic.tokens)} tokens, {saved.result['audio_seconds']:.1f}s) ---", flush=True)

    paths = read(ROOT / "models/paths.json")
    close_pipe = False
    if pipe is None:
        from lyra import YuE2Pipeline
        pipe = YuE2Pipeline.from_pretrained(
            paths["model"],
            vae=paths["vae"],
            precision="8bit",
            local_files_only=True,
            require_ac=True,
            generation_config=GenerationConfig(ode_steps=8),
        )
        close_pipe = True
    else:
        pipe.generation_config = GenerationConfig(ode_steps=8)

    monitor = ResourceMonitor(interval=0.1, require_ac=True, report_path=dest / "resources.json")
    with monitor:
        t0 = time.perf_counter()
        latents = pipe.synthesize(saved.semantic, noise=saved.noise)
        nar_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        audio = pipe.decode(latents)
        vae_seconds = time.perf_counter() - t1

    flac_path = dest / "audio.flac"
    sf.write(str(flac_path), audio, 48000, format="FLAC")

    abc_seconds = saved.semantic.plan.timing.get("seconds", 0.0)
    semantic_seconds = saved.semantic.timing.get("seconds", 0.0)
    transcription_seconds = parent_measurement.get("transcription_seconds", 0.0)
    load_seconds = parent_measurement.get("load_seconds", 0.0)
    audio_seconds = len(audio) / 48000
    e2e_seconds = abc_seconds + semantic_seconds + nar_seconds + vae_seconds + transcription_seconds + load_seconds
    rtf = e2e_seconds / audio_seconds
    speed_multiplier = audio_seconds / e2e_seconds

    resources = monitor.report()
    timing = {
        "abc": saved.semantic.plan.timing,
        "semantic": saved.semantic.timing,
        "nar_seconds": round(nar_seconds, 2),
        "vae_seconds": round(vae_seconds, 2),
        "load_seconds": round(load_seconds, 2),
        "transcription_seconds": round(transcription_seconds, 2),
        "e2e_seconds": round(e2e_seconds, 2),
        "32step_nar_seconds": round(parent_measurement.get("timing", {}).get("nar_seconds", 0.0), 2),
        "nar_speedup": round(parent_measurement.get("timing", {}).get("nar_seconds", 0.0) / nar_seconds, 2) if nar_seconds > 0 else 0,
    }

    receipt = {
        "id": case_id,
        "mode": "fast_8step",
        "ode_steps": 8,
        "status": "complete",
        "audio_seconds": round(audio_seconds, 2),
        "e2e_seconds": round(e2e_seconds, 2),
        "rtf": round(rtf, 3),
        "speed_multiplier": round(speed_multiplier, 2),
        "timing": timing,
        "audio_sha256": digest(flac_path),
        "audio_quality": {
            "sample_rate": 48000,
            "channels": 2,
            "finite": bool(np.isfinite(audio).all()),
            "rms": float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
            "peak": float(np.abs(audio).max()),
            "clipping_fraction": float(np.mean(np.abs(audio) >= 1.0)),
        },
        "resources": {k: v for k, v in resources.items() if k != "samples"},
    }
    write(dest / "measurement.json", receipt)
    print(f"✓ {case_id} 8-step complete: Audio={audio_seconds:.1f}s, NAR={nar_seconds:.1f}s (32-step: {timing['32step_nar_seconds']}s, {timing['nar_speedup']}x faster), E2E={e2e_seconds:.1f}s, RTF={rtf:.3f} ({speed_multiplier:.2f}x)", flush=True)

    if close_pipe:
        pipe.close()

    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()

    manifest = read(REPORT / "cases.json")
    cases = manifest["cases"]

    from lyra import YuE2Pipeline
    from yue2.protocol import GenerationConfig
    paths = read(ROOT / "models/paths.json")

    print("=== Initializing MLX Pipeline for 8-step Fast Mode Synthesis ===", flush=True)
    t_init = time.perf_counter()
    with YuE2Pipeline.from_pretrained(
        paths["model"],
        vae=paths["vae"],
        precision="8bit",
        local_files_only=True,
        require_ac=True,
        generation_config=GenerationConfig(ode_steps=8),
    ) as pipe:
        print(f"Pipeline initialized in {time.perf_counter() - t_init:.2f}s\n", flush=True)
        results = []
        for c in cases:
            cid = c["id"]
            if args.only and cid not in args.only:
                continue
            res = run_case_8step(cid, pipe=pipe)
            results.append(res)

    write(REPORT / "fast_8step_measurements.json", results)
    print(f"\nAll {len(results)} cases generated in 8-step fast mode successfully!", flush=True)


if __name__ == "__main__":
    main()

