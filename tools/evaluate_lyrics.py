"""Local ASR lyric-adherence proxy; run in the separately installed MOSS environment.

This is not SongBench PER or a perceptual music-quality score. The reference
lyrics are used only for scoring and are never given to the recognizer.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf


def units(text, chinese):
    text = re.sub(r"\[[^\]]*\]", "", text).lower()
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text) if chinese else re.findall(r"[a-z0-9]+", text)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def audio_checks(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    block = round(rate * 0.1)
    rms = np.sqrt(np.mean(audio[:len(audio) // block * block].reshape(-1, block, 2) ** 2, axis=(1, 2)))
    return {"sample_rate": rate, "channels": audio.shape[1], "finite": bool(np.isfinite(audio).all()),
            "peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio ** 2))),
            "near_full_scale_fraction": float(np.mean(np.abs(audio) >= 0.999)),
            "quiet_100ms_fraction_below_minus_60db": float(np.mean(rms < 0.001))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    from moss_transcribe_diarize.mlx.model import load_model
    from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript

    model = load_model(str(args.model), strict=True)
    results = []
    for record in json.loads(args.comparison.read_text()):
        folder = Path(record["directory"])
        request = json.loads((folder / "request.json").read_text())
        result = model.generate(folder / "audio.flac", max_tokens=4096, temperature=0.0)
        segments = subtitle_segments_from_transcript(result.text, postprocess=False)
        transcript = " ".join(s.text for s in segments)
        chinese = bool(re.search(r"[\u3400-\u9fff]", request["lyrics"]))
        ref, hyp = units(request["lyrics"], chinese), units(transcript, chinese)
        data = {"case": record["case"], "precision": record["precision"],
                "raw_transcript": result.text, "transcript": transcript,
                "segments": [s.to_dict() for s in segments], "reference_units": len(ref),
                "recognized_units": len(hyp), "edit_distance": edit_distance(ref, hyp),
                "asr_error_rate": edit_distance(ref, hyp) / len(ref),
                "metric": "Chinese characters plus Latin words" if chinese else "English WER",
                "asr_max_tokens_reached": result.generation_tokens >= 4096,
                "audio_checks": audio_checks(folder / "audio.flac"),
                "asr_model": str(args.model.resolve()), "asr_seconds": result.total_time}
        results.append(data)
        (args.comparison.parent / "lyric-quality.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: data[k] for k in ("case", "precision", "asr_error_rate", "transcript")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
