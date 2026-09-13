"""Whole-song native MLX transcription with official overlap and ABC export."""
from pathlib import Path
import subprocess
import time

import mlx.core as mx
import numpy as np

from .model import SheetSage2
from .upstream.tokenization_sheetsage2 import SheetSage2Tokenizer
from .upstream.generation_sheetsage2 import (
    PromptGrammarState, build_overlap_prefix_tokens, decode_generated_tokens,
    event_time_map, stitched_window_events,
)
from .upstream.windows import sliding_window_plan
from .upstream.exports_sheetsage2 import export_result
from yue2.storage import sha256_file, write_json


def generate_tokens(model, tokenizer, waveform, prompts, prefix=None, stop_time=None,
                    cancelled=None, progress=None):
    prefix = list(prefix) if prefix is not None else tokenizer.prompt_prefix(prompts)
    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]
    if len(prefix) >= model.config["max_output_seq_len"]:
        raise ValueError("Overlap prefix fills the decoder context")
    state = PromptGrammarState(tokenizer)
    for token in prefix[prefix.index(tokenizer.out_token) + 1:]:
        state.update(token)
    memory = model.encode(waveform, cancelled)
    mx.eval(memory)
    cache, ids, tokens = None, prefix, list(prefix)
    ended = False
    while len(tokens) < model.config["max_output_seq_len"]:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during transcription decoding")
        logits, cache = model.decode(memory, ids, cache)
        allowed = mx.array(state.allowed(None))
        token = int(mx.argmax(mx.where(allowed, logits[0, -1], -mx.inf)).item())
        tokens.append(token)
        ended = state.update(token)
        if not ended and stop_time is not None and tokenizer.time_token_start <= token < tokenizer.time_token_end:
            if tokenizer.token_to_time_id(token) / tokenizer.time_hz >= stop_time:
                tokens.append(tokenizer.eos_token)
                ended = True
        if ended:
            break
        ids = [token]
        if progress is not None and len(tokens) % 100 == 0:
            progress({"stage": "decoding", "tokens": len(tokens)})
        mx.eval(cache)
    truncated = not ended
    if tokens[-1] != tokenizer.eos_token:
        tokens.append(tokenizer.eos_token)
    return np.asarray(tokens, dtype=np.int64), truncated


def transcribe(audio, output, *, model=None, model_path="m-a-p/SheetSage2",
               base_model="m-a-p/MERT-v2-FullSong", offline=False, cache_dir=None,
               task="melody-full", preset="default", max_seconds=None, cancelled=None,
               progress=None):
    source, output = Path(audio), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Transcription needs a fresh output directory")
    if task not in {"full", "melody-full", "melody-vocal"} or preset not in {"default", "paper"}:
        raise ValueError("Invalid transcription task or window preset")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive; omit to process the complete recording")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source.resolve()), "-vn"]
    if max_seconds is not None:
        command += ["-t", str(max_seconds)]
    command += ["-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1"]
    decoded_audio = subprocess.run(command, capture_output=True, timeout=600, check=True)
    waveform = np.frombuffer(decoded_audio.stdout, dtype="<f4")
    if len(waveform) < 1025 or not np.isfinite(waveform).all():
        raise ValueError("Audio must contain at least 1025 finite samples at 24 kHz")
    start = time.perf_counter()
    if model is None:
        model = SheetSage2.from_pretrained(model_path, base_model, offline=offline, cache_dir=cache_dir)
    cfg = model.config
    tokenizer = SheetSage2Tokenizer(cfg["input_audio_length"], cfg["time_hz"], cfg["tokenizer_schema_version"],
                                    expected_fingerprint=cfg["tokenizer_fingerprint"])
    prompts = ["timestamp", "downbeat_meter", "structure", "key"]
    prompts += ["chord_full", "melody_full"] if task == "full" else ["melody_vocal" if task == "melody-vocal" else "melody_full"]
    prompts = tokenizer.normalize_prompts(prompts)
    duration, length = len(waveform) / 24000, cfg["input_audio_length"]
    windows = sliding_window_plan(duration, length, 200 if preset == "default" else 100,
                                   100 if preset == "default" else 0)
    stitched, records, warnings = [], [], []
    if preset == "paper":
        warnings.append("Paper window/export preset with FFmpeg resampling; not a reproduction of the torchaudio paper frontend")
    output.mkdir(parents=True, exist_ok=True)
    for index, window in enumerate(windows):
        if progress:
            progress({"stage": "encoding", "window": index + 1, "windows": len(windows)})
        prefix, base = None, 0
        if index:
            _, prefix, base = build_overlap_prefix_tokens(stitched, tokenizer, prompts, window["start"], window["prefix_end"])
            limit = cfg["max_output_seq_len"] - (16 if preset == "paper" else 128)
            if prefix is not None and len(prefix) >= limit:
                if preset == "paper":
                    prefix = None
                    warnings.append("Overlap prefix exceeded the paper context")
                else:
                    raise ValueError("Overlap prefix fills context; transcription cannot continue faithfully")
        offset = round(window["start"] * 24000)
        segment = waveform[offset:offset + round(length * 24000)]
        stop_time = None if preset == "paper" else (window["generation_stop"] if window["generation_stop"] is not None else min(duration - window["start"], length))
        tokens, truncated = generate_tokens(model, tokenizer, segment, prompts, prefix, stop_time, cancelled, progress)
        decoded, warning = decode_generated_tokens(tokenizer, tokens, source.stem, index)
        if warning:
            warnings.append(warning)
        stitched.extend(stitched_window_events(decoded, event_time_map(decoded, length), window["start"],
                        window["accept_start"], window["accept_end"], duration, index, base or 0))
        if preset == "paper":
            stitched.sort(key=lambda e: (float(e.get("time", 0)), int(e.get("window_index", 0)),
                                        int(e.get("source_subbeat", e["subbeat"]))))
        records.append({**window, "tokens": tokens.tolist(), "truncated": truncated})
        write_json(output / "windows.json", records)
    if preset != "paper":
        stitched.sort(key=lambda e: (e["time"], e["global_subbeat"]))
    decoded = dict(schema_version=tokenizer.schema_version, prompts=list(prompts), events=stitched, has_eos=True)
    exported = export_result(decoded, tokenizer, output, duration, paper=preset == "paper", melody_only=task != "full")
    payload = exported.pop("payload")
    result = {**exported, "backend": "mlx", "dtype": "float32", "model": model.identity,
              "source_audio_sha256": sha256_file(source), "duration_seconds": duration,
              "task": task, "preset": preset, "audio_decode": "ffmpeg_float32_mono_24000",
              "warnings": warnings, "truncated": any(r["truncated"] for r in records),
              "elapsed_seconds": time.perf_counter() - start}
    result["status"] = "complete" if payload.get("abc") and not result.get("abc_error") else "failed"
    write_json(output / "result.json", result)
    return {**result, **payload}
