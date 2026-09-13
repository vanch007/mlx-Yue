"""Pinned YuE2 checkpoint acquisition and strict MLX artifact conversion."""

from __future__ import annotations

import contextlib
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import tempfile
from typing import BinaryIO, Iterable, Mapping

MODEL_REPO = "m-a-p/YuE2-3B"
MODEL_REVISION = "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
VAE_REPO = "m-a-p/YuE2-Vae"
VAE_REVISION = "95535e72a97bc0f09b8ada125d26b4009428c0e8"
UPSTREAM_COMMIT = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"
UPSTREAM_REPOSITORY = "https://github.com/multimodal-art-projection/YuE"

_FORMAT = "lyra-yue2-mlx"
_QUANTIZATION_MODE = "affine"
_QUANTIZATION_GROUP_SIZE = 64
_MLX_VERSION = "0.32.2"
_MLX_LM_VERSION = "0.31.3"
_SAFETENSORS_VERSION = "0.7.0"
_BUFFER_SIZE = 8 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")

# These are the complete inference inputs fetched from the pinned snapshots. The
# non-weight hashes are pinned too: a local directory is never attributed to a
# Hub revision merely because a caller supplied (or omitted) a revision string.
_GENERATOR_SOURCE_FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "bytes": 959,
        "sha256": "ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329",
    },
    "qwen.tiktoken": {
        "bytes": 2_561_218,
        "sha256": "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186",
    },
    "weights_manifest.json": {
        "bytes": 179,
        "sha256": "2296f82c29dcb11aeefde07e216403a32fc86c8f6da2ec15a70102abfd936a93",
    },
    "model.safetensors": {
        "bytes": 7_261_441_640,
        "sha256": "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9",
    },
    "LICENSE": {
        "bytes": 20_309,
        "sha256": "060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0",
    },
    "THIRD_PARTY_NOTICES.md": {
        "bytes": 793,
        "sha256": "14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3",
    },
    "licenses/SnakeBeta-NVIDIA-MIT.txt": {
        "bytes": 1_076,
        "sha256": "da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc",
    },
    "licenses/stable-audio-tools-MIT.txt": {
        "bytes": 1_069,
        "sha256": "a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101",
    },
}

_VAE_SOURCE_FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "bytes": 1_378,
        "sha256": "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970",
    },
    "weights_manifest.json": {
        "bytes": 178,
        "sha256": "017d64a4d288217a43fcac3866a0e1d55f796fa891d4dfb6de0bd170ab750026",
    },
    "model.safetensors": {
        "bytes": 530_512_720,
        "sha256": "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346",
    },
    "LICENSE": _GENERATOR_SOURCE_FILES["LICENSE"],
    "THIRD_PARTY_NOTICES.md": _GENERATOR_SOURCE_FILES["THIRD_PARTY_NOTICES.md"],
    "licenses/SnakeBeta-NVIDIA-MIT.txt": _GENERATOR_SOURCE_FILES[
        "licenses/SnakeBeta-NVIDIA-MIT.txt"
    ],
    "licenses/stable-audio-tools-MIT.txt": _GENERATOR_SOURCE_FILES[
        "licenses/stable-audio-tools-MIT.txt"
    ],
}

_COPY_FILES = (
    "config.json",
    "qwen.tiktoken",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/SnakeBeta-NVIDIA-MIT.txt",
    "licenses/stable-audio-tools-MIT.txt",
)
_BASE_OUTPUT_FILES = frozenset((*_COPY_FILES, "ar-bf16.safetensors", "nar-bf16.safetensors"))
_PRECISIONS = frozenset({"bf16", "8bit", "4bit"})
_DTYPE_BYTES = {"BF16": 2, "U32": 4}

_EXPECTED_CONFIG = {
    "model_type": "yue2",
    "architectures": ["YuE2ForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 184704,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "max_position_embeddings": 24576,
    "tie_word_embeddings": False,
    "latent_type": "vae",
    "latent_dim": 64,
    "max_latent_frames": 24576,
    "timestep_shift": 1.0,
}

_CONVERSION_SETTINGS = {
    "generator_dtype": "bfloat16",
    "ar_extraction": "yue2.fast.ar_keys",
    "nar_partition": "all source tensors not selected by ar_keys",
    "positional_buffer_dtype": "bfloat16",
    "quantization": {
        "mode": _QUANTIZATION_MODE,
        "group_size": _QUANTIZATION_GROUP_SIZE,
        "modules": "mlx.nn.Linear",
        "linear_only": True,
        "embedding_dtype": "bfloat16",
    },
    "versions": {
        "mlx": _MLX_VERSION,
        "mlx_lm": _MLX_LM_VERSION,
        "safetensors": _SAFETENSORS_VERSION,
    },
}


class _TensorSpec:
    __slots__ = ("dtype", "shape", "start", "end")

    def __init__(self, dtype: str, shape: tuple[int, ...], start: int, end: int):
        self.dtype = dtype
        self.shape = shape
        self.start = start
        self.end = end

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class _DigestWriter:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data).cast("B")
        try:
            while view:
                chunk = view[:_BUFFER_SIZE]
                written = self.stream.write(chunk)
                if written is None:
                    written = len(chunk)
                if written <= 0:
                    raise OSError("Checkpoint write made no progress")
                self.digest.update(chunk[:written])
                self.bytes += written
                view = view[written:]
        finally:
            view.release()

    def record(self) -> dict[str, object]:
        return {"bytes": self.bytes, "sha256": self.digest.hexdigest()}


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _same_json(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _decode_json(data: bytes, description: str) -> object:
    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description}") from error


def _read_json(path: Path, description: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size > _MAX_JSON_BYTES:
        raise ValueError(f"{description} is unexpectedly large")
    return _decode_json(path.read_bytes(), description)


def _write_json(path: Path, value: object) -> None:
    data = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _safe_name(name: object) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Manifest contains an unsafe filename")
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or pure.as_posix() != name
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"Manifest contains an unsafe filename: {name!r}")
    return name


def _require_exact_mapping(value: object, keys: set[str], description: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Invalid {description}")
    return value


def _verify_pinned_tree(
    directory: Path,
    expected: Mapping[str, Mapping[str, object]],
    *,
    hash_weights: bool,
) -> dict[str, dict[str, object]]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    weights = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.safetensors")
        if path.is_file()
    }
    if weights != {"model.safetensors"}:
        raise ValueError(
            f"Pinned checkpoint needs exactly model.safetensors; found {sorted(weights)}"
        )

    identity: dict[str, dict[str, object]] = {}
    for name, pinned in expected.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned checkpoint file is missing: {name}")
        size = path.stat().st_size
        if size != pinned["bytes"]:
            raise ValueError(f"Pinned checkpoint size mismatch: {name}")
        if name != "model.safetensors" or hash_weights:
            digest = _sha256_file(path)
            if digest != pinned["sha256"]:
                raise ValueError(f"Pinned checkpoint content mismatch: {name}")
        else:
            digest = str(pinned["sha256"])
        identity[name] = {"bytes": size, "sha256": digest}

    weight_manifest = _require_exact_mapping(
        _read_json(directory / "weights_manifest.json", "source weight manifest"),
        {"schema", "files"},
        "source weight manifest",
    )
    expected_weights = {"model.safetensors": expected["model.safetensors"]}
    if (
        type(weight_manifest["schema"]) is not int
        or weight_manifest["schema"] != 1
        or not _same_json(weight_manifest["files"], expected_weights)
    ):
        raise ValueError("Source weight manifest does not identify the pinned checkpoint")
    return identity


def _validate_generator_config(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        raise ValueError("Generator config must be a JSON object")
    for key, expected in _EXPECTED_CONFIG.items():
        if key not in config or not _same_json(config[key], expected):
            raise ValueError(f"Pinned generator config mismatch: {key}")
    if config.get("dtype", config.get("torch_dtype")) != "bfloat16":
        raise ValueError("Pinned generator must declare bfloat16 weights")
    return config


def _expected_shapes(config: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab = int(config["vocab_size"])
    latent = int(config["latent_dim"])
    frames = int(config["max_latent_frames"])

    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
        "llm2vae.weight": (latent, hidden),
        "llm2vae.bias": (latent,),
        "vae2llm.weight": (hidden, latent),
        "vae2llm.bias": (hidden,),
        "time_embedder.mlp.0.weight": (hidden, 256),
        "time_embedder.mlp.0.bias": (hidden,),
        "time_embedder.mlp.2.weight": (hidden, hidden),
        "time_embedder.mlp.2.bias": (hidden,),
        "latent_pos_embed.pe": (frames, hidden),
    }
    attention = {
        "q_proj.weight": (heads * head_dim, hidden),
        "k_proj.weight": (kv_heads * head_dim, hidden),
        "v_proj.weight": (kv_heads * head_dim, hidden),
        "o_proj.weight": (hidden, heads * head_dim),
        "q_norm.weight": (head_dim,),
        "k_norm.weight": (head_dim,),
    }
    mlp = {
        "gate_proj.weight": (intermediate, hidden),
        "up_proj.weight": (intermediate, hidden),
        "down_proj.weight": (hidden, intermediate),
    }
    for index in range(layers):
        prefix = f"model.layers.{index}."
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)
        shapes[prefix + "nar_input_layernorm.weight"] = (hidden,)
        shapes[prefix + "nar_pre_mlp_layernorm.weight"] = (hidden,)
        for name, shape in attention.items():
            shapes[prefix + "self_attn." + name] = shape
            shapes[prefix + "nar_self_attn." + name] = shape
        for name, shape in mlp.items():
            shapes[prefix + "mlp." + name] = shape
            shapes[prefix + "nar_mlp." + name] = shape
    return shapes


def _ar_keys(config: Mapping[str, object]) -> frozenset[str]:
    # This import is deliberately local: conversion uses the pinned upstream
    # extraction specification without introducing an AR-loader import cycle.
    from yue2.fast import ar_keys

    return frozenset(ar_keys(config))


def _linear_keys(config: Mapping[str, object]) -> frozenset[str]:
    keys = {"lm_head.weight"}
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{index}."
        keys.update(
            prefix + name
            for name in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
    return frozenset(keys)


def _read_safetensors_header(path: Path) -> tuple[dict[str, _TensorSpec], dict[str, str], int]:
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f"Invalid safetensors file: {path.name}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not (2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES):
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        if 8 + header_size > size:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        header_bytes = stream.read(header_size)
    header = _decode_json(header_bytes, f"safetensors header in {path.name}")
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path.name}")

    raw_metadata = header.pop("__metadata__", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise ValueError(f"Invalid safetensors metadata in {path.name}")
    metadata = dict(raw_metadata)

    specs: dict[str, _TensorSpec] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, raw in header.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError(f"Invalid tensor entry in {path.name}")
        if set(raw) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor metadata for {name}")
        dtype = raw["dtype"]
        shape = raw["shape"]
        offsets = raw["data_offsets"]
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported tensor dtype {dtype!r} for {name}")
        if not isinstance(shape, list) or not shape or any(
            type(dimension) is not int or dimension <= 0 for dimension in shape
        ):
            raise ValueError(f"Invalid tensor shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"Invalid tensor offsets for {name}")
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(f"Tensor byte size does not match shape for {name}")
        spec = _TensorSpec(dtype, tuple(shape), offsets[0], offsets[1])
        specs[name] = spec
        ranges.append((spec.start, spec.end, name))

    position = 0
    for start, end, name in sorted(ranges):
        if start != position:
            raise ValueError(f"Non-contiguous or overlapping tensor data near {name}")
        position = end
    if position != size - 8 - header_size:
        raise ValueError(f"Safetensors payload length mismatch: {path.name}")
    return specs, metadata, 8 + header_size


def _validate_specs(
    specs: Mapping[str, _TensorSpec],
    expected: Mapping[str, tuple[int, ...]],
    *,
    dtypes: Mapping[str, str] | None = None,
) -> None:
    if set(specs) != set(expected):
        raise ValueError(
            "Tensor set mismatch: "
            f"missing={sorted(set(expected) - set(specs))}, "
            f"unexpected={sorted(set(specs) - set(expected))}"
        )
    for name, shape in expected.items():
        spec = specs[name]
        if spec.shape != shape:
            raise ValueError(
                f"Tensor shape mismatch for {name}: expected {shape}, found {spec.shape}"
            )
        dtype = "BF16" if dtypes is None else dtypes[name]
        if spec.dtype != dtype:
            raise ValueError(
                f"Tensor dtype mismatch for {name}: expected {dtype}, found {spec.dtype}"
            )


def _verify_source_model(
    source: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, tuple[int, ...]]]:
    identity = _verify_pinned_tree(source, _GENERATOR_SOURCE_FILES, hash_weights=True)
    config = _validate_generator_config(_read_json(source / "config.json", "generator config"))
    expected = _expected_shapes(config)
    specs, _, _ = _read_safetensors_header(source / "model.safetensors")
    _validate_specs(specs, expected)

    ar = _ar_keys(config)
    if not ar or not ar < set(expected):
        raise ValueError("Upstream AR extraction did not partition the complete model")
    if "model.norm.weight" not in ar or "latent_pos_embed.pe" in ar:
        raise ValueError("Upstream AR extraction violates the shared AR/NAR boundary")
    if _linear_keys(config) - ar:
        raise ValueError("AR Linear extraction is incomplete")
    return config, identity, expected


def _tensor_header(
    ordered: Iterable[tuple[str, str, tuple[int, ...]]], metadata: Mapping[str, str]
) -> tuple[bytes, dict[str, _TensorSpec]]:
    header: dict[str, object] = {"__metadata__": dict(metadata)}
    specs: dict[str, _TensorSpec] = {}
    offset = 0
    for name, dtype, shape in ordered:
        if name in specs:
            raise ValueError(f"Duplicate output tensor: {name}")
        length = math.prod(shape) * _DTYPE_BYTES[dtype]
        spec = _TensorSpec(dtype, shape, offset, offset + length)
        specs[name] = spec
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [spec.start, spec.end],
        }
        offset += length
    encoded = json.dumps(
        header, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded, specs


def _copy_bytes(source: BinaryIO, destination: _DigestWriter, length: int) -> None:
    remaining = length
    while remaining:
        block = source.read(min(_BUFFER_SIZE, remaining))
        if not block:
            raise ValueError("Source safetensors data ended unexpectedly")
        destination.write(block)
        remaining -= len(block)


def _write_partition(
    source: Path,
    destination: Path,
    names: frozenset[str],
    *,
    partition: str,
) -> dict[str, object]:
    source_specs, _, data_start = _read_safetensors_header(source)
    if not names <= set(source_specs):
        raise ValueError(f"Missing tensors while writing {partition} partition")
    ordered_names = sorted(names, key=lambda name: source_specs[name].start)
    header, _ = _tensor_header(
        ((name, source_specs[name].dtype, source_specs[name].shape) for name in ordered_names),
        {"format": "mlx", "partition": partition, "precision": "bfloat16"},
    )
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        writer = _DigestWriter(output_stream)
        writer.write(header)
        for name in ordered_names:
            spec = source_specs[name]
            input_stream.seek(data_start + spec.start)
            _copy_bytes(input_stream, writer, spec.nbytes)
        output_stream.flush()
        os.fsync(output_stream.fileno())
        return writer.record()


def _quantized_expected(
    ar_expected: Mapping[str, tuple[int, ...]],
    linear_keys: frozenset[str],
    bits: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    expected: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for name, shape in ar_expected.items():
        if name not in linear_keys:
            expected[name] = shape
            dtypes[name] = "BF16"
            continue
        if len(shape) != 2 or shape[1] % _QUANTIZATION_GROUP_SIZE:
            raise ValueError(f"Linear weight cannot use group-64 quantization: {name}")
        output, input_ = shape
        if input_ * bits % 32:
            raise ValueError(f"Linear weight cannot be packed at {bits} bits: {name}")
        prefix = name[: -len("weight")]
        expected[name] = (output, input_ * bits // 32)
        expected[prefix + "scales"] = (output, input_ // _QUANTIZATION_GROUP_SIZE)
        expected[prefix + "biases"] = (output, input_ // _QUANTIZATION_GROUP_SIZE)
        dtypes[name] = "U32"
        dtypes[prefix + "scales"] = "BF16"
        dtypes[prefix + "biases"] = "BF16"
    return expected, dtypes


def _require_quantizer_versions() -> None:
    from importlib.metadata import PackageNotFoundError, version

    for package, expected in (
        ("mlx", _MLX_VERSION),
        ("mlx-lm", _MLX_LM_VERSION),
        ("safetensors", _SAFETENSORS_VERSION),
    ):
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"{package} is required for AR quantization") from error
        if actual != expected:
            raise RuntimeError(
                f"AR quantization requires {package}=={expected}; found {actual}"
            )


def _write_mlx_array(writer: _DigestWriter, array: object) -> None:
    view = memoryview(array).cast("B")
    try:
        writer.write(view)
    finally:
        view.release()


def _write_quantized_ar(
    source: Path,
    destination: Path,
    config: Mapping[str, object],
    bits: int,
) -> dict[str, object]:
    _require_quantizer_versions()
    import mlx.core as mx
    from .measure import GPUExecution

    source_specs, _, data_start = _read_safetensors_header(source)
    ar_keys = _ar_keys(config)
    all_expected = _expected_shapes(config)
    ar_expected = {name: all_expected[name] for name in ar_keys}
    _validate_specs(source_specs, ar_expected)
    linear_keys = _linear_keys(config)
    expected, dtypes = _quantized_expected(ar_expected, linear_keys, bits)

    ordered_source = sorted(ar_keys, key=lambda name: source_specs[name].start)
    ordered_output: list[tuple[str, str, tuple[int, ...]]] = []
    for name in ordered_source:
        if name in linear_keys:
            prefix = name[: -len("weight")]
            for output_name in (name, prefix + "scales", prefix + "biases"):
                ordered_output.append((output_name, dtypes[output_name], expected[output_name]))
        else:
            ordered_output.append((name, "BF16", ar_expected[name]))
    header, _ = _tensor_header(
        ordered_output,
        {
            "format": "mlx",
            "partition": "ar",
            "precision": f"{bits}bit",
            "quantization": "affine-group64-linear",
        },
    )

    with (
        GPUExecution(backend="mlx") as guard,
        source.open("rb") as raw_source,
        destination.open("xb") as output_stream,
    ):
        weights = mx.load(str(source))
        if set(weights) != ar_keys:
            raise ValueError("MLX loaded a different AR tensor set than the safetensors header")
        writer = _DigestWriter(output_stream)
        writer.write(header)
        for name in ordered_source:
            guard.check()
            spec = source_specs[name]
            if name not in linear_keys:
                raw_source.seek(data_start + spec.start)
                _copy_bytes(raw_source, writer, spec.nbytes)
                del weights[name]
                continue

            weight = weights.pop(name)
            packed, scales, biases = mx.quantize(
                weight,
                group_size=_QUANTIZATION_GROUP_SIZE,
                bits=bits,
                mode=_QUANTIZATION_MODE,
                stream=mx.gpu,
            )
            mx.eval(packed, scales, biases)
            prefix = name[: -len("weight")]
            arrays = (
                (name, packed, mx.uint32),
                (prefix + "scales", scales, mx.bfloat16),
                (prefix + "biases", biases, mx.bfloat16),
            )
            for output_name, array, dtype in arrays:
                if tuple(array.shape) != expected[output_name] or array.dtype != dtype:
                    raise RuntimeError(f"MLX emitted incompatible quantization data for {name}")
                _write_mlx_array(writer, array)
            del weight, packed, scales, biases, arrays, array
            mx.clear_cache()
            guard.check()
        if weights:
            raise RuntimeError("Not every BF16 AR tensor was consumed during quantization")
        output_stream.flush()
        os.fsync(output_stream.fileno())
        record = writer.record()

    specs, metadata, _ = _read_safetensors_header(destination)
    _validate_specs(specs, expected, dtypes=dtypes)
    if metadata != {
        "format": "mlx",
        "partition": "ar",
        "precision": f"{bits}bit",
        "quantization": "affine-group64-linear",
    }:
        raise RuntimeError("Quantized AR metadata was not preserved")
    return record


def _precision_record(precision: str) -> dict[str, object]:
    if precision == "bf16":
        return {
            "ar": "ar-bf16.safetensors",
            "nar": "nar-bf16.safetensors",
            "dtype": "bfloat16",
        }
    bits = int(precision.removesuffix("bit"))
    return {
        "ar": f"ar-{precision}.safetensors",
        "bits": bits,
        "group_size": _QUANTIZATION_GROUP_SIZE,
        "mode": _QUANTIZATION_MODE,
        "linear_only": True,
        "embedding_dtype": "bfloat16",
    }


def _new_manifest(
    source_identity: Mapping[str, Mapping[str, object]],
    precisions: Iterable[str],
    files: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    ordered_precisions = ["bf16", *(name for name in ("8bit", "4bit") if name in precisions)]
    return {
        "schema": 1,
        "format": _FORMAT,
        "source": {
            "repository": MODEL_REPO,
            "revision": MODEL_REVISION,
            "revision_proof": "verified-pinned-content",
            "identity": {"files": copy.deepcopy(dict(source_identity))},
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        },
        "conversion": copy.deepcopy(_CONVERSION_SETTINGS),
        "precisions": {name: _precision_record(name) for name in ordered_precisions},
        "files": copy.deepcopy(dict(files)),
    }


def _actual_tree(directory: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError(f"Converted artifacts cannot contain symlinks: {relative}")
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise ValueError(f"Unsupported converted artifact: {relative}")
    return files, directories


def _validate_source_manifest(source: object) -> Mapping[str, object]:
    source = _require_exact_mapping(
        source,
        {"repository", "revision", "revision_proof", "identity"},
        "source provenance",
    )
    if (
        source["repository"] != MODEL_REPO
        or source["revision"] != MODEL_REVISION
        or source["revision_proof"] != "verified-pinned-content"
    ):
        raise ValueError("Converted source provenance is not the pinned YuE2 generator")
    identity = _require_exact_mapping(source["identity"], {"files"}, "source identity")
    records = identity["files"]
    if not isinstance(records, dict) or set(records) != set(_GENERATOR_SOURCE_FILES):
        raise ValueError("Converted source identity has missing or unexpected files")
    for name, pinned in _GENERATOR_SOURCE_FILES.items():
        record = _require_exact_mapping(records[name], {"bytes", "sha256"}, f"source file {name}")
        if not _same_json(record, pinned):
            raise ValueError(f"Converted source identity mismatch: {name}")
    return source


def _validate_file_records(files: object) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(files, dict):
        raise ValueError("Converted file manifest must be an object")
    validated: dict[str, Mapping[str, object]] = {}
    for raw_name, raw_record in files.items():
        name = _safe_name(raw_name)
        record = _require_exact_mapping(raw_record, {"bytes", "sha256"}, f"file record {name}")
        if type(record["bytes"]) is not int or record["bytes"] <= 0:
            raise ValueError(f"Invalid byte count for {name}")
        if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"]):
            raise ValueError(f"Invalid SHA256 for {name}")
        validated[name] = record
    return validated


def _validate_conversion(
    directory: Path,
    manifest: object,
    *,
    verify_hashes: bool,
) -> dict[str, object]:
    manifest = _require_exact_mapping(
        manifest,
        {"schema", "format", "source", "upstream", "conversion", "precisions", "files"},
        "conversion manifest",
    )
    if (
        type(manifest["schema"]) is not int
        or manifest["schema"] != 1
        or manifest["format"] != _FORMAT
    ):
        raise ValueError("Unsupported conversion manifest schema or format")
    _validate_source_manifest(manifest["source"])
    if not _same_json(
        manifest["upstream"],
        {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        },
    ):
        raise ValueError("Converted artifact uses an incompatible upstream implementation")
    if not _same_json(manifest["conversion"], _CONVERSION_SETTINGS):
        raise ValueError("Converted artifact settings are incompatible with this runtime")

    precisions = manifest["precisions"]
    if (
        not isinstance(precisions, dict)
        or not set(precisions) <= _PRECISIONS
        or "bf16" not in precisions
    ):
        raise ValueError("Converted precision manifest is invalid")
    for name, value in precisions.items():
        if not _same_json(value, _precision_record(name)):
            raise ValueError(f"Converted precision settings are incompatible: {name}")

    expected_files = set(_BASE_OUTPUT_FILES)
    for name in ("8bit", "4bit"):
        if name in precisions:
            expected_files.add(f"ar-{name}.safetensors")
    files = _validate_file_records(manifest["files"])
    if set(files) != expected_files:
        raise ValueError(
            "Converted file manifest mismatch: "
            f"missing={sorted(expected_files - set(files))}, "
            f"unexpected={sorted(set(files) - expected_files)}"
        )

    actual_files, actual_directories = _actual_tree(directory)
    allowed_optional = {"README.md"}
    required_files = expected_files | {"conversion.json"}
    if not (required_files <= actual_files <= required_files | allowed_optional):
        raise ValueError(
            "Converted directory has missing or unexpected files: "
            f"missing={sorted(required_files - actual_files)}, "
            f"unexpected={sorted(actual_files - (required_files | allowed_optional))}"
        )
    expected_directories = {
        PurePosixPath(name).parent.as_posix()
        for name in expected_files
        if PurePosixPath(name).parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise ValueError("Converted directory has missing or unexpected subdirectories")

    for name, record in files.items():
        path = directory / name
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Converted file size mismatch: {name}")
        if verify_hashes and _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Converted file integrity failed: {name}")

    source_files = manifest["source"]["identity"]["files"]
    for name in _COPY_FILES:
        if files[name] != source_files[name]:
            raise ValueError(f"Copied pinned file identity changed: {name}")

    config = _validate_generator_config(_read_json(directory / "config.json", "generator config"))
    expected = _expected_shapes(config)
    ar_keys = _ar_keys(config)
    ar_expected = {name: expected[name] for name in ar_keys}
    nar_expected = {name: shape for name, shape in expected.items() if name not in ar_keys}

    ar_specs, ar_metadata, _ = _read_safetensors_header(directory / "ar-bf16.safetensors")
    _validate_specs(ar_specs, ar_expected)
    if ar_metadata != {"format": "mlx", "partition": "ar", "precision": "bfloat16"}:
        raise ValueError("BF16 AR safetensors metadata is incompatible")
    nar_specs, nar_metadata, _ = _read_safetensors_header(directory / "nar-bf16.safetensors")
    _validate_specs(nar_specs, nar_expected)
    if nar_metadata != {"format": "mlx", "partition": "nar", "precision": "bfloat16"}:
        raise ValueError("BF16 NAR safetensors metadata is incompatible")

    linear_keys = _linear_keys(config)
    for name in ("8bit", "4bit"):
        if name not in precisions:
            continue
        bits = int(name.removesuffix("bit"))
        quant_expected, quant_dtypes = _quantized_expected(ar_expected, linear_keys, bits)
        specs, metadata, _ = _read_safetensors_header(directory / f"ar-{name}.safetensors")
        _validate_specs(specs, quant_expected, dtypes=quant_dtypes)
        if metadata != {
            "format": "mlx",
            "partition": "ar",
            "precision": name,
            "quantization": "affine-group64-linear",
        }:
            raise ValueError(f"{name} AR safetensors metadata is incompatible")

    return dict(manifest)


def verify_conversion(directory: str | os.PathLike[str]) -> dict[str, object]:
    """Verify the complete converted artifact and return its schema-1 manifest."""
    path = Path(directory).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"Converted model directory does not exist: {path}")
    path = path.resolve()
    manifest_path = path / "conversion.json"
    if manifest_path.is_symlink():
        raise ValueError("conversion.json cannot be a symlink")
    manifest = _read_json(manifest_path, "conversion manifest")
    return _validate_conversion(path, manifest, verify_hashes=True)


def fetch_models(
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
) -> tuple[Path, Path]:
    """Fetch only the pinned YuE2-3B generator and default YuE2-Vae snapshots."""
    if type(local_files_only) is not bool:
        raise TypeError("local_files_only must be a bool")
    from huggingface_hub import snapshot_download

    cache = None if cache_dir is None else str(Path(cache_dir).expanduser())
    generator = Path(
        snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            cache_dir=cache,
            local_files_only=local_files_only,
            allow_patterns=sorted(_GENERATOR_SOURCE_FILES),
        )
    ).resolve()
    vae = Path(
        snapshot_download(
            VAE_REPO,
            revision=VAE_REVISION,
            cache_dir=cache,
            local_files_only=local_files_only,
            allow_patterns=sorted(_VAE_SOURCE_FILES),
        )
    ).resolve()
    # Hub verifies downloaded blobs. Recheck every small pinned file and the
    # exact LFS sizes here; prepare/load perform the full weight hashes.
    _verify_pinned_tree(generator, _GENERATOR_SOURCE_FILES, hash_weights=False)
    _verify_pinned_tree(vae, _VAE_SOURCE_FILES, hash_weights=False)
    return generator, vae


@contextlib.contextmanager
def _conversion_lock(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.conversion.lock"
    with lock_path.open("a+b") as stream:
        try:
            import fcntl
        except ImportError as error:
            raise RuntimeError("Checkpoint conversion requires POSIX file locking") from error
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_exchange(left: Path, right: Path) -> None:
    encoded_left, encoded_right = os.fsencode(left), os.fsencode(right)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(encoded_left, encoded_right, 0x00000002)  # RENAME_SWAP
    elif sys.platform.startswith("linux"):
        function = libc.renameat2
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(-100, encoded_left, -100, encoded_right, 0x00000002)
    else:
        raise RuntimeError("Atomic incremental conversion is unsupported on this platform")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), f"{left} <-> {right}")


def _atomic_install(source: Path, destination: Path) -> None:
    encoded_source, encoded_destination = os.fsencode(source), os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(
            encoded_source, encoded_destination, 0x00000004  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        function = libc.renameat2
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(-100, encoded_source, -100, encoded_destination, 0x00000001)
    else:
        raise RuntimeError("Atomic checkpoint installation is unsupported on this platform")
    if result:
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"Conversion destination appeared while preparing: {destination}"
            )
        raise OSError(code, os.strerror(code), f"{source} -> {destination}")


def _copy_pinned_files(source: Path, destination: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name in _COPY_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        records[name] = _file_record(target)
    return records


def _build_fresh(
    source: Path,
    staging: Path,
    config: Mapping[str, object],
    source_identity: Mapping[str, Mapping[str, object]],
    precision: str,
) -> None:
    records = _copy_pinned_files(source, staging)
    expected = _expected_shapes(config)
    ar_keys = _ar_keys(config)
    records["ar-bf16.safetensors"] = _write_partition(
        source / "model.safetensors",
        staging / "ar-bf16.safetensors",
        ar_keys,
        partition="ar",
    )
    records["nar-bf16.safetensors"] = _write_partition(
        source / "model.safetensors",
        staging / "nar-bf16.safetensors",
        frozenset(set(expected) - ar_keys),
        partition="nar",
    )
    precisions = {"bf16"}
    if precision != "bf16":
        bits = int(precision.removesuffix("bit"))
        name = f"ar-{precision}.safetensors"
        records[name] = _write_quantized_ar(
            staging / "ar-bf16.safetensors", staging / name, config, bits
        )
        precisions.add(precision)
    manifest = _new_manifest(source_identity, precisions, records)
    _write_json(staging / "conversion.json", manifest)
    _validate_conversion(staging, manifest, verify_hashes=False)


def _link_existing(
    source: Path,
    staging: Path,
    files: Iterable[str],
) -> None:
    for name in files:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / name, target, follow_symlinks=False)


def _add_quantized_variant(
    output: Path,
    manifest: dict[str, object],
    config: Mapping[str, object],
    precision: str,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incremental.", dir=output.parent))
    swapped = False
    try:
        existing_files = set(manifest["files"])
        _link_existing(output, staging, existing_files)
        name = f"ar-{precision}.safetensors"
        bits = int(precision.removesuffix("bit"))
        record = _write_quantized_ar(
            staging / "ar-bf16.safetensors", staging / name, config, bits
        )
        updated = copy.deepcopy(manifest)
        updated["precisions"][precision] = _precision_record(precision)
        updated["files"][name] = record
        _write_json(staging / "conversion.json", updated)
        _validate_conversion(staging, updated, verify_hashes=False)
        _atomic_exchange(staging, output)
        swapped = True
        _fsync_directory(output.parent)
    finally:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                if not swapped:
                    raise


def prepare(
    model_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    precision: str = "bf16",
    source_revision: str | None = None,
) -> Path:
    """Convert the exact pinned generator, preserving BF16 and adding one AR precision."""
    if precision not in _PRECISIONS:
        raise ValueError("precision must be one of: bf16, 8bit, 4bit")
    if source_revision is not None and source_revision != MODEL_REVISION:
        raise ValueError(f"Generator revision must be pinned to {MODEL_REVISION}")

    source = Path(model_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError("Converted output directory cannot be a symlink")
    output = raw_output.resolve(strict=False)
    if output == source or output.is_relative_to(source):
        raise ValueError("Converted output must be outside the source checkpoint")

    with _conversion_lock(output):
        config, source_identity, _ = _verify_source_model(source)
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise FileExistsError(f"Conversion destination is not a model directory: {output}")
            manifest = verify_conversion(output)
            if manifest["source"]["identity"]["files"] != source_identity:
                raise ValueError("Existing conversion came from a different source identity")
            if precision in manifest["precisions"]:
                return output
            _add_quantized_variant(output, manifest, config, precision)
            return output

        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.fresh.", dir=output.parent))
        try:
            _build_fresh(source, staging, config, source_identity, precision)
            _atomic_install(staging, output)
            _fsync_directory(output.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return output


__all__ = [
    "MODEL_REPO",
    "MODEL_REVISION",
    "VAE_REPO",
    "VAE_REVISION",
    "UPSTREAM_COMMIT",
    "fetch_models",
    "prepare",
    "verify_conversion",
]
