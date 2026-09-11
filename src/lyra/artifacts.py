"""Integrity-checked stage replay; never reconstruct score IDs from decoded text."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import soundfile as sf

from yue2.pipeline import SemanticResult, SymbolicPlan
from yue2.protocol import CODEC_SIZE, GenerationConfig
from yue2.storage import identity, sha256_file, verify_result, write_json


@dataclass
class SavedSong:
    semantic: SemanticResult
    latents: np.ndarray
    noise: np.ndarray | None
    config: dict
    result: dict


def save_plan_artifacts(plan, directory, generation_config):
    if not isinstance(plan, SymbolicPlan):
        raise TypeError("plan must be a SymbolicPlan")
    if not isinstance(generation_config, GenerationConfig):
        raise TypeError("generation_config must be a GenerationConfig")
    directory = Path(directory)
    plan.save(directory)
    plan_path = directory / "plan.json"
    data = json.loads(plan_path.read_text())
    data["generation_config"] = generation_config.to_dict()
    write_json(plan_path, data)
    manifest_path = directory / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["plan.json"] = sha256_file(plan_path)
    write_json(manifest_path, manifest)


def load_plan_artifacts(directory):
    directory = Path(directory)
    plan = SymbolicPlan.load(directory)
    data = json.loads((directory / "plan.json").read_text())
    saved = data.get("generation_config")
    if saved is None:
        config_path = directory / "config.json"
        if not config_path.is_file():
            return plan, None
        config = json.loads(config_path.read_text())
        saved = config.get("generation") if isinstance(config, dict) else None
    if not isinstance(saved, dict):
        raise ValueError("Saved plan generation configuration is invalid")
    try:
        generation_config = GenerationConfig.from_dict(saved)
    except (TypeError, ValueError) as error:
        raise ValueError("Saved plan generation configuration is invalid") from error
    return plan, generation_config


def load_artifacts(directory) -> SavedSong:
    directory = Path(directory)
    result = verify_result(directory)
    plan = SymbolicPlan.load(directory)
    request = json.loads((directory / "request.json").read_text())
    if request != plan.request.to_dict():
        raise ValueError("Saved plan and song request disagree")
    config = json.loads((directory / "config.json").read_text())
    if not isinstance(config, dict) or not isinstance(config.get("generation"), dict):
        raise ValueError("Saved song lacks its effective generation configuration")
    try:
        GenerationConfig.from_dict(config["generation"])
    except (TypeError, ValueError) as error:
        raise ValueError("Saved generation configuration is invalid") from error
    weights = result.get("weights")
    if not isinstance(weights, dict) or not {"mot", "vae"} <= weights.keys():
        raise ValueError("Saved song lacks model weight identities")
    expected_identity = identity({"request": request, "config": config, "weights": weights})
    if result.get("identity") != expected_identity:
        raise ValueError("Saved result identity disagrees with its request, configuration or weights")

    tokens = np.load(directory / "semantic.npy", allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or not len(tokens):
        raise ValueError("Expected nonempty integer semantic token vector")
    if tokens.min() < 0 or tokens.max() >= CODEC_SIZE:
        raise ValueError("Saved semantic tokens are outside the codec vocabulary")
    latents = np.load(directory / "latent.npy", allow_pickle=False)
    if latents.shape != (len(tokens), 64) or latents.dtype != np.float32 or not np.isfinite(latents).all():
        raise ValueError("Saved latents violate the finite FP32 [T,64] contract")
    noise_path = directory / "noise.npy"
    has_noise = "noise.npy" in result["artifacts"]
    if noise_path.is_file() != has_noise:
        raise ValueError("Saved solver noise and its artifact manifest disagree")
    noise = None
    if has_noise:
        noise = np.load(noise_path, allow_pickle=False)
        if noise.shape != latents.shape or noise.dtype != np.float32 or not np.isfinite(noise).all():
            raise ValueError("Saved noise violates the finite FP32 [T,64] contract")

    timing = result.get("timing")
    truncated = result.get("truncated")
    if (
        not isinstance(timing, dict)
        or not isinstance(timing.get("semantic"), dict)
        or timing.get("abc") != plan.timing
    ):
        raise ValueError("Saved result lacks consistent stage timing")
    if (
        not isinstance(truncated, dict)
        or type(truncated.get("abc")) is not bool
        or type(truncated.get("semantic")) is not bool
        or truncated["abc"] != plan.truncated
    ):
        raise ValueError("Saved result lacks consistent truncation metadata")

    try:
        audio = sf.info(directory / "audio.flac")
    except RuntimeError as error:
        raise ValueError("Saved audio is unreadable") from error
    expected_frames = 1920 * len(tokens) - 64
    if audio.samplerate != 48000 or audio.channels != 2 or audio.frames != expected_frames:
        raise ValueError("Saved audio violates the natural 48 kHz stereo length contract")
    if (
        result.get("sample_rate") != audio.samplerate
        or result.get("audio_seconds") != audio.frames / audio.samplerate
    ):
        raise ValueError("Saved audio metadata disagrees with the audio artifact")

    semantic = SemanticResult(plan, tokens.tolist(), timing["semantic"], truncated["semantic"])
    return SavedSong(semantic, latents, noise, config, result)
