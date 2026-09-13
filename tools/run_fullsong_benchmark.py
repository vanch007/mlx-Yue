"""Serial full-song benchmarks with measured resources and source provenance.

Each case runs in a fresh process. Failed/truncated attempts are retained and
never silently rerolled. Resume validates the saved input and audio hashes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/official-fullsong"
OUT = ROOT / "outputs/official_fullsong"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def batch_exit_code(receipts):
    return int(not receipts or any(r.get("status") != "complete" or
               set(r.get("truncated", {})) != {"abc", "semantic"} or
               any(r["truncated"].values()) for r in receipts))


def attempt_directory(cid, number=1):
    base = OUT / cid
    return base if number == 1 else base / f"attempt-{number}"


def attempt_number(cid):
    base = OUT / cid
    numbers = [1] if base.exists() else []
    numbers += [int(p.name.split('-')[1]) for p in base.glob('attempt-*') if p.is_dir()]
    return max(numbers, default=0)


def latest_directory(cid):
    return attempt_directory(cid, max(1, attempt_number(cid)))


def measurement_for(cid):
    destination = latest_directory(cid)
    if not (destination / 'measurement.json').exists():
        return None
    receipt = read(destination / 'measurement.json')
    receipt['artifact_directory'] = str(destination.relative_to(ROOT))
    receipt['prior_attempts'] = []
    for number in range(1, attempt_number(cid)):
        previous = attempt_directory(cid, number) / 'measurement.json'
        old = read(previous)
        receipt['prior_attempts'].append({
            'attempt': number, 'status': old['status'], 'error': old.get('error'),
            'receipt_path': str(previous.relative_to(ROOT)), 'receipt_sha256': digest(previous),
        })
    return receipt


def validate_case(case):
    request, source = case["request"], case["source"]
    if request["style"] != source["tags"] or request["lyrics"] != source["lyrics"]:
        raise ValueError("Official lyrics/style mismatch")
    if case["score_mode"] == "supplied" and request.get("abc") != source["abc"]:
        raise ValueError("Official ABC mismatch")
    if "semantic_sampling" in request or "abc_sampling" in request:
        raise ValueError("Do not override native generation budgets in this benchmark")
    if digest(ROOT / case["official_audio_path"]) != case["official_audio_sha256"]:
        raise ValueError("Reference audio identity mismatch")


def worker(case, attempt=1):
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from lyra import YuE2Pipeline
    from lyra.measure import GPUExecution, ResourceMonitor
    from yue2.protocol import GenerationConfig

    validate_case(case)
    dest = attempt_directory(case['id'], attempt)
    dest.mkdir(exist_ok=False)
    request = dict(case["request"])
    paths = read(ROOT / "models/paths.json")
    write(dest / "input.json", case)
    receipt = dict(id=case["id"], status="failed", case_sha256=hashlib.sha256(
        json.dumps(case, sort_keys=True).encode()).hexdigest())
    receipt.update(attempt=attempt, benchmark_script_sha256=digest(Path(__file__)),
                   retry_policy='Only interrupted/failed execution, identical input and seed; no quality reroll')
    monitor = ResourceMonitor(interval=0.1, require_ac=True,
                              report_path=dest / "resources.json")
    try:
        with monitor:
            begin = time.perf_counter()
            transcription_seconds = 0
            if case["score_mode"] == "transcribed":
                from lyra.transcription.pipeline import transcribe
                if digest(ROOT / case["source_audio_path"]) != case["source_audio_sha256"]:
                    raise ValueError("Cover source identity mismatch")
                with GPUExecution(require_ac=True) as guard:
                    def cancelled():
                        guard.check()
                        return False
                    tr = transcribe(ROOT / case["source_audio_path"], dest / "transcription",
                                    offline=True, cache_dir=str(ROOT / "models/hf-cache"),
                                    task="full", cancelled=cancelled)
                if tr["status"] != "complete" or tr["truncated"]:
                    raise ValueError("Incomplete transcription")
                request["abc"] = (dest / "transcription/score.abc").read_text()
                transcription_seconds = time.perf_counter() - begin
                gc.collect()
                mx.clear_cache()
            load_start = time.perf_counter()
            with YuE2Pipeline.from_pretrained(
                paths["model"], vae=paths["vae"], precision="8bit", local_files_only=True,
                require_ac=True, generation_config=GenerationConfig(),
            ) as pipe:
                load_seconds = time.perf_counter() - load_start
                song = pipe(**request)
                generation_end = time.perf_counter()
                song.save_artifacts(dest / "song")
            saved_end = time.perf_counter()
            info = sf.info(dest / "song/audio.flac")
            audio = song.audio
            e2e = generation_end - begin
            receipt.update(status="truncated" if any(song.truncated.values()) else "complete",
                           truncated=song.truncated, audio_seconds=info.duration,
                           official_seconds=case["official_seconds"],
                           duration_ratio=info.duration / case["official_seconds"],
                           e2e_seconds=e2e, rtf=e2e / info.duration,
                           speed_multiplier=info.duration / e2e,
                           load_seconds=load_seconds, transcription_seconds=transcription_seconds,
                           save_seconds=saved_end - generation_end, timing=song.timing,
                           configuration=song.config, identity=song.request_identity,
                           audio_sha256=digest(dest / "song/audio.flac"),
                           result_sha256=digest(dest / "song/result.json"),
                           audio_quality=dict(sample_rate=info.samplerate, channels=info.channels,
                                              finite=bool(np.isfinite(audio).all()),
                                              rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                                              peak=float(np.abs(audio).max()),
                                              clipping_fraction=float(np.mean(np.abs(audio) >= 1.0))),
                           musical_quality_acceptance="pending human listening; not established by duration or EOS")
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    resources = monitor.report()
    receipt["resources"] = {k: v for k, v in resources.items() if k != "samples"}
    write(dest / "measurement.json", receipt)
    print(json.dumps({k: receipt.get(k) for k in ("id", "status", "audio_seconds", "e2e_seconds", "rtf", "error")}), flush=True)
    return receipt["status"] != "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    cases = read(REPORT / "cases.json")["cases"]
    if args.worker:
        return worker(next(c for c in cases if c["id"] == args.worker), args.attempt)
    for case in cases:
        if args.only and case["id"] not in args.only:
            continue
        validate_case(case)
        dest = latest_directory(case['id'])
        attempt = max(1, attempt_number(case['id']))
        if dest.exists():
            receipt = read(dest / "measurement.json")
            expected = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
            if receipt["case_sha256"] != expected:
                raise ValueError("Saved input identity mismatch")
            if receipt.get("audio_sha256") and digest(dest / "song/audio.flac") != receipt["audio_sha256"]:
                raise ValueError("Saved audio identity mismatch")
            if not args.retry_failed or receipt['status'] != 'failed':
                print("Preserving existing attempt", case["id"], receipt["status"], flush=True)
                continue
            attempt += 1
            dest = attempt_directory(case['id'], attempt)
        from lyra.measure import power_source
        if not power_source()['ac_connected']:
            print('STOP: Connect AC power before resuming benchmarks', flush=True)
            return 1
        print("START", case["id"], flush=True)
        suffix = '' if attempt == 1 else f'-attempt-{attempt}'
        with (OUT / (case["id"] + suffix + ".log")).open("x") as log:
            result = subprocess.run([sys.executable, __file__, "--worker", case["id"],
                                     '--attempt', str(attempt)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print("FINISH", case["id"], "exit", result.returncode, flush=True)
        if not (dest / "measurement.json").exists():
            raise RuntimeError(f"Worker crashed; inspect {dest}")
        receipts = [receipt for c in cases if (receipt := measurement_for(c['id'])) is not None]
        write(REPORT / "measurements.json", receipts)
    selected = [case for case in cases if not args.only or case["id"] in args.only]
    return batch_exit_code([measurement_for(case['id']) or {'status': 'missing'} for case in selected])


if __name__ == "__main__":
    sys.exit(main())
