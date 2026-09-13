"""Capture real checkpoint evidence in the isolated pinned Torch environment.

Run with PYTHONPATH=src:vendor/yue/src .oracle/bin/python tools/oracle.py ...
The oracle never imports MLX or the production model implementation.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from lyra.measure import GPUExecution
from yue2.pipeline import YuE2Pipeline, SymbolicPlan
from yue2.protocol import (
    CODEC_OFFSET, CODEC_SIZE, CONTEXT, GenerationConfig, SongRequest,
    token_prefixes, negative_prefix,
)
from yue2.storage import identity, model_identity, write_json, sha256_file, verify_result
from yue2.tokenization_yue2 import YuE2TextTokenizer

MODEL_REVISION = "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
VAE_REVISION = "95535e72a97bc0f09b8ada125d26b4009428c0e8"
GENERATED_ABC_BUDGET = 1024
MAX_REFERENCE_FRAMES = 500
SOLVER_STEPS = 32


def _atomic_save(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_savez(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resource_evidence(output):
    result = {}
    for name in ("resources.jsonl", "resources.json"):
        path = Path(output) / name
        if path.is_file():
            result[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _guard_cancelled(guard):
    guard.check()
    return False


def _invocation(args):
    inputs = {}
    direct = {
        "corpus": args.corpus,
        "abc": args.abc,
        "latents": args.latents,
        "states_from": args.states_from,
    }
    for name, value in direct.items():
        if value is not None and Path(value).is_file():
            inputs[name] = sha256_file(value)
    if args.source is not None:
        source = Path(args.source)
        for name in ("result.json", "plan.json", "semantic.npy", "config.json"):
            path = source / name
            if path.is_file():
                inputs[f"source/{name}"] = sha256_file(path)
    for label, root in (("model", args.model), ("vae", args.vae)):
        directory = Path(root)
        for name in ("config.json", "weights_manifest.json", "yue2_generation_config.json"):
            path = directory / name
            if path.is_file():
                inputs[f"{label}/{name}"] = sha256_file(path)
    record = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "script_sha256": sha256_file(__file__),
        "input_sha256": inputs,
        "upstream_commit": "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e",
        "model_revision": MODEL_REVISION,
        "vae_revision": VAE_REVISION,
        "runtime": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            **{name: version(name) for name in ("torch", "transformers", "numpy")},
        },
        "capture_config": {
            "semantic_frame_budget": args.max_tokens if args.action == "generate" else None,
            "generated_abc_token_budget": (
                GENERATED_ABC_BUDGET if args.action == "generate" else None
            ),
            "supplied_abc_is_exact": args.abc is not None,
            "ar_prefill_chunk_size": args.prefill_chunk_size if args.action == "ar" else None,
            "nar_max_frames": MAX_REFERENCE_FRAMES if args.action == "nar" else None,
            "nar_solver_steps": SOLVER_STEPS if args.action == "nar" else None,
            "memory_budget_gib": args.memory_budget_gib,
            "resource_backend": "mps",
            "generator_dtype": "bfloat16",
            "vae_dtype": "float32",
        },
    }
    record["invocation_sha256"] = identity(record)
    return record


def noise_for(frames, seed):
    return torch.randn(
        (frames, 64),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def request_for(args):
    corpus = json.loads(Path(args.corpus).read_text())
    source = next(case for case in corpus["cases"] if case["id"] == args.case)
    request = {key: source[key] for key in SongRequest.__dataclass_fields__ if key in source}
    if args.mode is not None:
        request["cot"] = args.mode
    if args.abc is not None:
        request["abc"] = Path(args.abc).read_bytes().decode("utf-8")
    return SongRequest(**request)


def make_model(path):
    from yue2.modeling_yue2 import YuE2ForCausalLM

    return YuE2ForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("mps")


def generate(args, guard):
    config = GenerationConfig()
    request = request_for(args)
    semantic_sampling = {
        "max_tokens": args.max_tokens,
        "min_tokens": min(200, args.max_tokens),
    }
    abc_sampling = (
        {"max_tokens": GENERATED_ABC_BUDGET}
        if request.abc is None and request.cot != "off"
        else None
    )
    partial_path = args.output / "generate.partial.json"
    generation_progress = {
        "status": "running",
        "abc_output_tokens": 0,
        "semantic_output_tokens": 0,
        "semantic_frame_budget": args.max_tokens,
        "generated_abc_token_budget": GENERATED_ABC_BUDGET if abc_sampling else None,
    }
    write_json(partial_path, generation_progress)

    def cancelled():
        return _guard_cancelled(guard)

    def on_token(phase, _token):
        guard.check()
        key = f"{phase}_output_tokens"
        generation_progress[key] = generation_progress.get(key, 0) + 1
        if generation_progress[key] == 1 or generation_progress[key] % 16 == 0:
            write_json(partial_path, generation_progress)

    guard.check()
    with YuE2Pipeline(
        args.model,
        args.vae,
        device="mps",
        backend="torch-eager",
        generation_config=config,
        memory_budget_gib=args.memory_budget_gib,
        progress=True,
    ) as pipe:
        song = pipe(
            **request.to_dict(),
            abc_sampling=abc_sampling,
            semantic_sampling=semantic_sampling,
            cancelled=cancelled,
            on_token=on_token,
        )
        guard.check()
    if request.abc is not None and song.abc != request.abc:
        raise AssertionError("Supplied ABC changed during reference generation")
    _atomic_save(
        args.output / "noise.npy",
        noise_for(len(song.semantic.tokens), request.seed).numpy(),
    )
    summary = {
        "audio_seconds": len(song.audio) / song.sample_rate,
        "semantic_frames": len(song.semantic.tokens),
        "semantic_frame_budget": args.max_tokens,
        "generated_abc_token_budget": GENERATED_ABC_BUDGET if abc_sampling else None,
        "supplied_abc_exact": request.abc is not None,
        "truncated": song.truncated,
        "timing": song.timing,
    }
    write_json(args.output / "generate.json", summary)
    partial_path.unlink(missing_ok=True)
    return summary, song


def _ar_inputs(args, tokenizer):
    maximum = max(args.lengths)
    if args.source is None:
        request = request_for(args)
        prefix = token_prefixes(request, tokenizer)
        if args.branch == "negative":
            abc_ids = tokenizer.encode(request.abc) if request.abc is not None else []
            prefix = negative_prefix(request, tokenizer, abc_ids)
        trace = (prefix * ((maximum + len(prefix) - 1) // len(prefix)))[:maximum]
        return trace, {
            "kind": "repeated_prefix_stress",
            "branch": args.branch,
            "prefix_tokens": len(prefix),
        }

    source = Path(args.source)
    plan = SymbolicPlan.load(source)
    if token_prefixes(plan.request, tokenizer, plan.abc_ids) != plan.prefix:
        raise ValueError("Saved plan prefix disagrees with request/exact ABC IDs")
    prefix = (
        negative_prefix(plan.request, tokenizer, plan.abc_ids)
        if args.branch == "negative" else plan.prefix
    )
    trace = list(prefix)
    semantic_path = source / "semantic.npy"
    frames = 0
    if semantic_path.is_file():
        verify_result(source)
        codec = np.load(semantic_path, allow_pickle=False)
        if codec.ndim != 1 or codec.dtype.kind not in "iu" or not codec.size:
            raise ValueError("Expected nonempty integer semantic token vector")
        if codec.min() < 0 or codec.max() >= CODEC_SIZE:
            raise ValueError("Saved semantic tokens are outside the codec vocabulary")
        frames = len(codec)
        trace.extend(int(token) + CODEC_OFFSET for token in codec)
    if maximum > len(trace):
        raise ValueError(
            f"AR trace has {len(trace)} tokens; cannot capture length {maximum} "
            "without inventing a continuation"
        )
    return trace[:maximum], {
        "kind": "saved_song_semantic" if frames else "saved_plan_prefix",
        "branch": args.branch,
        "source": str(source),
        "prefix_tokens": len(prefix),
        "semantic_frames": frames,
        "available_tokens": len(trace),
    }


def capture_ar(args, guard):
    from yue2.modeling_yue2 import StaticKVCache

    tokenizer = YuE2TextTokenizer(Path(args.model) / "qwen.tiktoken")
    trace, input_sequence = _ar_inputs(args, tokenizer)
    guard.check()
    weights = model_identity(args.model)
    guard.check()
    model = make_model(args.model)
    # AR capture never calls the acoustic branch. Release its unused weights so
    # long-context reference traces fit the same budget without changing AR math.
    for layer in model.model.layers:
        for name in ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp"):
            getattr(layer, name).to_empty(device="meta")
    for name in ("vae2llm", "llm2vae", "time_embedder", "latent_pos_embed"):
        getattr(model, name).to_empty(device="meta")
    torch.mps.synchronize()
    torch.mps.empty_cache()
    def clear_layer_workspace(_module, _inputs, _output):
        torch.mps.synchronize()
        torch.mps.empty_cache()

    # Long original-MPS captures accumulate unused workspaces within a single
    # prefill across 28 layers. Reclaim them without changing tensor arithmetic.
    cache_hooks = [layer.register_forward_hook(clear_layer_workspace) for layer in model.model.layers]
    outputs, metadata = {}, []
    partial_npz = args.output / "ar.partial.npz"
    partial_json = args.output / "ar.partial.json"
    try:
        guard.check()
        for length in args.lengths:
            write_json(
                partial_json,
                {
                    "status": "running",
                    "completed_cases": metadata,
                    "active_length": length,
                    "prefill_chunk_size": args.prefill_chunk_size,
                },
            )
            ids = trace[:length]
            cache = StaticKVCache(
                model.config.num_hidden_layers,
                1,
                model.config.num_key_value_heads,
                length,
                model.config.head_dim,
                next(model.parameters()).dtype,
                torch.device("mps"),
            )
            start_time = time.perf_counter()
            prefill_count = length - 4
            try:
                with torch.inference_mode():
                    logits = None
                    for start in range(0, prefill_count, args.prefill_chunk_size):
                        guard.check()
                        stop = min(start + args.prefill_chunk_size, prefill_count)
                        absolute = torch.arange(start, stop, device="mps", dtype=torch.long)
                        logits = model(
                            torch.tensor([ids[start:stop]], device="mps"),
                            position_ids=absolute[None],
                            cache_position=absolute,
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=1,
                        ).logits[0, -1]
                        guard.check()
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    if logits is None:
                        raise RuntimeError("AR fixture requires a nonempty prefill")
                    captured_logits = [logits.float().cpu().numpy()]
                    for token in ids[-4:]:
                        guard.check()
                        position = cache.get_seq_length()
                        absolute = torch.tensor([position], device="mps", dtype=torch.long)
                        logits = model(
                            torch.tensor([[token]], device="mps"),
                            position_ids=absolute[None],
                            cache_position=absolute,
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=1,
                        ).logits[0, -1]
                        captured_logits.append(logits.float().cpu().numpy())
                        guard.check()
                positions = np.unique(
                    np.concatenate(
                        (
                            np.arange(min(16, length)),
                            np.arange(max(0, length // 2 - 4), min(length, length // 2 + 4)),
                            np.arange(max(0, length - 16), length),
                        )
                    )
                )
                outputs[f"ids_{length}"] = np.asarray(ids, dtype=np.int32)
                outputs[f"logits_{length}"] = np.stack(captured_logits)
                outputs[f"positions_{length}"] = positions
                for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
                    index = torch.tensor(positions, device="mps")
                    outputs[f"k_{length}_{layer}"] = (
                        key.index_select(2, index).float().cpu().numpy()
                    )
                    outputs[f"v_{length}_{layer}"] = (
                        value.index_select(2, index).float().cpu().numpy()
                    )
                torch.mps.synchronize()
                guard.check()
                row = {
                    "length": length,
                    "seconds": time.perf_counter() - start_time,
                    "prefill_tokens": prefill_count,
                    "incremental_tokens": 4,
                    "prefill_chunk_size": args.prefill_chunk_size,
                    "prefill_chunks": (
                        prefill_count + args.prefill_chunk_size - 1
                    ) // args.prefill_chunk_size,
                    "position_ids": "absolute",
                    "key_visibility": "full_causal_history",
                }
                metadata.append(row)
                _atomic_savez(partial_npz, outputs)
                write_json(
                    partial_json,
                    {"status": "running", "completed_cases": metadata, "active_length": None},
                )
                print(json.dumps(row), flush=True)
            finally:
                del cache
                torch.mps.empty_cache()
        _atomic_savez(args.output / "ar.npz", outputs)
        result = {
            "cases": metadata,
            "model": weights,
            "dtype": "bfloat16",
            "storage_policy": "unused acoustic weights released; unused MPS buffers cleared after each AR layer and prefill chunk",
            "input_sequence": input_sequence,
            "prefill": {
                "chunk_size": args.prefill_chunk_size,
                "position_ids": "absolute",
                "key_visibility": "full_causal_history",
            },
        }
        write_json(args.output / "ar.json", result)
        partial_npz.unlink(missing_ok=True)
        partial_json.unlink(missing_ok=True)
        return {"cases": metadata, "dtype": "bfloat16", "input_sequence": input_sequence}
    finally:
        for hook in cache_hooks:
            hook.remove()
        del model
        torch.mps.empty_cache()


def capture_nar(args, guard):
    from yue2.nar import CachedNAR, song_chunks

    plan = SymbolicPlan.load(args.source)
    available = np.load(Path(args.source) / "semantic.npy", allow_pickle=False).tolist()
    codec = available[:args.frames]
    if not codec:
        raise ValueError("NAR fidelity fixture must contain at least one semantic frame")
    if len(codec) > MAX_REFERENCE_FRAMES:
        raise ValueError(f"NAR fidelity fixtures are limited to {MAX_REFERENCE_FRAMES} frames")
    chunks = song_chunks(plan.prefix, codec, plan.request.seed)
    if len(chunks) != 1:
        raise ValueError(
            "Fidelity fixture must fit one original chunk; "
            "use boundary regressions for multi-chunk"
        )
    guard.check()
    weights = model_identity(args.model)
    guard.check()
    model = make_model(args.model)
    engine = None
    partial_npz = args.output / "nar.partial.npz"
    partial_json = args.output / "nar.partial.json"
    try:
        engine = CachedNAR(model, chunks[0])
        guard.check()
        arrays = {
            "prefix": np.asarray(plan.prefix, dtype=np.int32),
            "codec": np.asarray(codec, dtype=np.int32),
            "noise": chunks[0].noise.numpy(),
        }
        for layer, (key, value) in enumerate(engine.cache):
            arrays[f"k_{layer}"] = key.permute(1, 0, 2)[None].float().cpu().numpy()
            arrays[f"v_{layer}"] = value.permute(1, 0, 2)[None].float().cpu().numpy()
        _atomic_savez(partial_npz, arrays)
        write_json(partial_json, {"status": "running", "frames": len(codec), "completed_steps": 0})
        state = chunks[0].noise.to("mps", next(model.parameters()).dtype)
        dt = 1.0 / SOLVER_STEPS
        with torch.inference_mode():
            for step in range(SOLVER_STEPS):
                t = 1.0 - step * dt
                raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
                raw_mid = torch.logit(
                    torch.tensor(t - dt / 2, dtype=torch.float64)
                ).clamp(-20, 20).item()
                guard.check()
                first = engine.velocity(state, raw)
                guard.check()
                mid = state - first * (dt / 2)
                guard.check()
                velocity = engine.velocity(mid, raw_mid)
                guard.check()
                if step in (0, 1, 15, 31):
                    arrays[f"state_{step}"] = state.float().cpu().numpy()
                    arrays[f"velocity_{step}"] = first.float().cpu().numpy()
                    arrays[f"mid_{step}"] = mid.float().cpu().numpy()
                    arrays[f"mid_velocity_{step}"] = velocity.float().cpu().numpy()
                    _atomic_savez(partial_npz, arrays)
                state = state - velocity * dt
                write_json(
                    partial_json,
                    {
                        "status": "running",
                        "frames": len(codec),
                        "completed_steps": step + 1,
                        "solver_steps": SOLVER_STEPS,
                    },
                )
        arrays["latents"] = state.float().cpu().numpy()
        _atomic_savez(partial_npz, arrays)
        write_json(
            partial_json,
            {
                "status": "running",
                "frames": len(codec),
                "completed_steps": SOLVER_STEPS,
                "solver_steps": SOLVER_STEPS,
                "phase": "fixed_state_velocities" if args.states_from else "finalizing",
            },
        )
        if args.states_from:
            with np.load(args.states_from, allow_pickle=False) as fixed, torch.inference_mode():
                for step in (0, 1, 15, 31):
                    t = 1.0 - step / SOLVER_STEPS
                    raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
                    raw_mid = torch.logit(
                        torch.tensor(t - 1 / (2 * SOLVER_STEPS), dtype=torch.float64)
                    ).clamp(-20, 20).item()
                    for name, state_name, raw_time in (
                        (f"fixed_velocity_{step}", f"state_{step}", raw),
                        (f"fixed_mid_velocity_{step}", f"mid_{step}", raw_mid),
                    ):
                        guard.check()
                        fixed_state = torch.from_numpy(fixed[state_name].copy()).to(
                            "mps", next(model.parameters()).dtype
                        )
                        arrays[name] = engine.velocity(fixed_state, raw_time).float().cpu().numpy()
                        guard.check()
        _atomic_savez(args.output / "nar.npz", arrays)
        result = {
            "frames": len(codec),
            "seed": plan.request.seed,
            "steps": SOLVER_STEPS,
            "dtype": "bfloat16",
            "model": weights,
        }
        write_json(args.output / "nar.json", result)
        partial_npz.unlink(missing_ok=True)
        partial_json.unlink(missing_ok=True)
        summary = {
            "frames": len(codec),
            "steps": SOLVER_STEPS,
            "latents_finite": bool(np.isfinite(arrays["latents"]).all()),
        }
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        if engine is not None:
            engine.close()
        del model
        torch.mps.empty_cache()


def decode(args, guard):
    from yue2.modeling_vae import YuE2VAE

    guard.check()
    weights = model_identity(args.vae)
    guard.check()
    model = YuE2VAE.from_pretrained(
        args.vae,
        decoder_only=True,
        device="mps",
        local_files_only=True,
    )
    try:
        if args.latents.endswith(".npz"):
            with np.load(args.latents, allow_pickle=False) as data:
                latent = data["latents"].copy()
        else:
            latent = np.load(args.latents, allow_pickle=False)
        z = torch.from_numpy(latent.T[None])
        outputs = {}
        for core in (64, 1024):
            guard.check()
            outputs[f"core_{core}"] = model.decode_tiled(
                z,
                core_frames=core,
                halo_frames=16,
                on_progress=lambda _completed, _total: guard.check(),
            ).numpy()
            guard.check()
            _atomic_savez(args.output / "vae.partial.npz", outputs)
        if z.shape[-1] <= 1100:
            guard.check()
            outputs["full"] = model.decode(z).cpu().numpy()
            guard.check()
        _atomic_savez(args.output / "vae.npz", outputs)
        write_json(
            args.output / "vae.json",
            {
                "frames": z.shape[-1],
                "natural_length": model.natural_output_length(z.shape[-1]),
                "required_halo": model.required_halo(1024),
                "decoder_dtype": "float32",
                "weights": weights,
            },
        )
        (args.output / "vae.partial.npz").unlink(missing_ok=True)
        return {"frames": z.shape[-1], "outputs": sorted(outputs)}
    finally:
        del model
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "ar", "nar", "vae"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", default="validation/corpus.json")
    parser.add_argument("--case", default="city_lights")
    parser.add_argument("--mode", choices=("full", "melody", "off"))
    parser.add_argument("--abc")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--require-ac", action="store_true")
    parser.add_argument("--memory-budget-gib", type=float, default=16)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--branch", choices=("positive", "negative"), default="positive")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Full-generator FP32 calibration is currently unsupported on MPS",
    )
    parser.add_argument(
        "--source",
        help="Saved song/plan for AR traces or NAR conditioning; AR otherwise repeats its prompt as a stress input",
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--latents")
    parser.add_argument(
        "--states-from",
        help="BF16 NAR capture supplying fixed velocity-evaluation states",
    )
    args = parser.parse_args()

    if args.action == "generate" and not 1 <= args.max_tokens <= MAX_REFERENCE_FRAMES:
        parser.error(
            f"--max-tokens must be in [1, {MAX_REFERENCE_FRAMES}] "
            "for reference generation"
        )
    if args.action == "ar" and args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be a positive integer")
    if args.action == "ar" and (
        len(set(args.lengths)) != len(args.lengths)
        or any(not 5 <= length <= CONTEXT for length in args.lengths)
    ):
        parser.error(f"--lengths must be unique integers in [5, {CONTEXT}]")
    if args.action == "ar" and args.source is not None and (
        args.mode is not None or args.abc is not None
    ):
        parser.error("AR --source uses the saved request and cannot be combined with --mode/--abc")
    if args.action == "nar" and not 1 <= args.frames <= MAX_REFERENCE_FRAMES:
        parser.error(f"--frames must be in [1, {MAX_REFERENCE_FRAMES}] for NAR fixtures")
    if args.dtype == "fp32":
        parser.error(
            "Full-generator FP32 execution on MPS is unsupported: the 3.58B combined model "
            "requires about 14.5 GB for weights before activations; use a separately bounded "
            "calibration method once one is defined"
        )
    if args.action == "nar" and args.source is None:
        parser.error("nar requires --source")
    if args.action == "vae" and args.latents is None:
        parser.error("vae requires --latents")
    if args.states_from is not None and args.action != "nar":
        parser.error("--states-from is only valid for nar")

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable implicit MPS CPU fallback for oracle captures")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") == "1":
        raise RuntimeError("Disable MPS fast math for oracle fidelity")
    torch.set_float32_matmul_precision("highest")
    if version("transformers") != "4.57.6" or version("torch") != "2.11.0":
        raise RuntimeError(
            "Run this oracle with the locked .oracle environment (PyTorch 2.11.0); "
            "older MPS two-pass attention can corrupt BF16 reference memory"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Oracle capture requires an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    invocation = _invocation(args)
    write_json(args.output / "invocation.json", invocation)
    status_path = args.output / "capture.json"
    write_json(
        status_path,
        {
            "status": "running",
            "action": args.action,
            "invocation_sha256": invocation["invocation_sha256"],
            "started_unix": time.time(),
        },
    )
    actions = {"generate": generate, "ar": capture_ar, "nar": capture_nar, "vae": decode}
    generated_song = None
    try:
        with GPUExecution(
            backend="mps",
            memory_budget_gib=args.memory_budget_gib,
            resource_path=args.output / "resources.jsonl",
            require_ac=args.require_ac,
        ) as guard:
            guard.check()
            result = actions[args.action](args, guard)
            if args.action == "generate":
                summary, generated_song = result
            else:
                summary = result
            guard.check()
        completion = {
            "status": "model_execution_complete" if generated_song is not None else "complete",
            "action": args.action,
            "invocation_sha256": invocation["invocation_sha256"],
            "summary": summary,
            "resources": _resource_evidence(args.output),
            "finished_unix": time.time(),
        }
        if generated_song is not None:
            completion["artifact_status"] = "result.json is authoritative"
        write_json(status_path, completion)
        if generated_song is not None:
            generated_song.save_artifacts(args.output)
    except BaseException as exc:
        write_json(
            status_path,
            {
                "status": "failed_incomplete",
                "action": args.action,
                "invocation_sha256": invocation["invocation_sha256"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "resources": _resource_evidence(args.output),
                "finished_unix": time.time(),
            },
        )
        raise
    if generated_song is not None:
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
