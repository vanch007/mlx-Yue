"""Native FP32 Oobleck decoder for the official YuE2 VAE checkpoints.

Architecture: vendor/yue/src/yue2/modeling_vae.py (Apache/MIT notices retained).
MLX channels-last convolution contract:
https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        return x + mx.sin(x * mx.exp(self.alpha)) ** 2 / (mx.exp(self.beta) + 1e-9)


def activation(channels, use_snake):
    return SnakeBeta(channels) if use_snake else nn.ELU()


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, use_snake):
        super().__init__()
        self.layers = [activation(channels, use_snake),
                       nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
                       activation(channels, use_snake), nn.Conv1d(channels, channels, 1)]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_snake):
        super().__init__()
        self.layers = [activation(cin, use_snake),
                       nn.ConvTranspose1d(cin, cout, 2 * stride, stride=stride,
                                          padding=math.ceil(stride / 2)),
                       *[ResidualUnit(cout, d, use_snake) for d in (1, 3, 9)]]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if any(config.get(k, False) for k in
               ("antialias_activation", "use_nearest_upsample", "use_filter")):
            raise ValueError("Unsupported decoder architecture")
        if config.get("snake_type", "vanilla") != "vanilla":
            raise ValueError("Only released vanilla SnakeBeta decoders are supported")
        c, mults = config["channels"], [1, *config["c_mults"]]
        strides, snake = config["strides"], config.get("use_snake", False)
        self.strides = tuple(reversed(strides))
        self.latent_dim = config["latent_dim"]
        self.layers = [nn.Conv1d(config["latent_dim"], mults[-1] * c, 7, padding=3)]
        for i in range(len(mults) - 1, 0, -1):
            self.layers.append(DecoderBlock(mults[i] * c, mults[i - 1] * c, strides[i - 1], snake))
        self.layers += [activation(c, snake),
                        nn.Conv1d(c, config["out_channels"], 7, padding=3, bias=False),
                        nn.Tanh() if config.get("final_tanh", True) else nn.Identity()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def output_length(self, frames):
        for stride in self.strides:
            frames = frames * stride - stride % 2
        return frames

    def decode(self, latents, *, core_frames=256, halo_frames=16, full=False,
               cancelled=None, on_progress=None):
        """CPU FP32 [T,C] -> CPU FP32 stereo, retaining exact halo/crop boundaries."""
        z = np.asarray(latents)
        if z.ndim == 3 and z.shape[:2] == (1, self.latent_dim):
            z = z[0].T
        if z.ndim != 2 or z.shape[1] != self.latent_dim or len(z) < 1:
            raise ValueError("Expected nonempty latents [T,C] or [1,C,T]")
        if not np.isfinite(z).all():
            raise ValueError("VAE latents contain non-finite values")
        if type(core_frames) is not int or core_frames < 1 or halo_frames < 16:
            raise ValueError("Require positive core_frames and at least 16 halo frames")
        if full:
            core_frames = len(z)
        ratio, total = math.prod(self.strides), self.output_length(len(z))
        pieces = []
        count = math.ceil(len(z) / core_frames)
        for index, start in enumerate(range(0, len(z), core_frames), 1):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during VAE decoding")
            end = min(len(z), start + core_frames)
            left, right = max(0, start - halo_frames), min(len(z), end + halo_frames)
            tile = self(mx.array(z[None, left:right], dtype=mx.float32))[0]
            offset = (start - left) * ratio
            length = min(end * ratio, total) - start * ratio
            pieces.append(np.array(tile[offset:offset + length]))
            if on_progress is not None:
                on_progress(index, count)
        audio = np.concatenate(pieces)
        if audio.shape != (total, 2) or not np.isfinite(audio).all():
            raise ValueError("VAE violated finite stereo output geometry")
        return audio


def decoder_weights(weights):
    """Fold official weight_norm dim=0 and transpose both convolution layouts."""
    result = {}
    for full_key, value in weights.items():
        if not full_key.startswith("decoder.") or full_key.endswith("weight_g"):
            continue
        key = full_key.removeprefix("decoder.")
        value = np.asarray(value, dtype=np.float32)
        if key.endswith("weight_v"):
            gain = np.asarray(weights[full_key[:-1] + "g"], dtype=np.float32)
            norm = np.sqrt(np.sum(value * value, axis=(1, 2), keepdims=True))
            if np.any(norm == 0):
                raise ValueError("Zero weight-normalization denominator")
            value = value * (gain / norm)
            # ConvTranspose is the second module directly inside each decoder block.
            parts = key.split(".")
            transposed = len(parts) == 5 and parts[2:4] == ["layers", "1"]
            value = value.transpose(1, 2, 0) if transposed else value.transpose(0, 2, 1)
            key = key.removesuffix("_v")
        result[key] = mx.array(value)
    return result


def load_decoder(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = OobleckDecoder(config["decoder_config"])
    with safe_open(directory / "model.safetensors", framework="numpy") as reader:
        weights = {k: reader.get_tensor(k) for k in reader.keys() if k.startswith("decoder.")}
    model.load_weights(list(decoder_weights(weights).items()), strict=True)
    mx.eval(model.parameters())
    return model
