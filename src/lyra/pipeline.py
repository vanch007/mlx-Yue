"""YuE2's public stage protocol with MLX generation and a native MLX FP32 VAE."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform
import shutil
import time

import mlx.core as mx
import numpy as np

from yue2.pipeline import (
    SemanticResult,
    SongResult as ReferenceSongResult,
    YuE2Pipeline as ReferencePipeline,
)
from yue2.progress import Progress
from yue2.protocol import CONTEXT, GenerationConfig, token_prefixes
from yue2.storage import identity, model_identity, resolve_model, sha256_file, write_json
from yue2.tokenization_yue2 import YuE2TextTokenizer

from .conversion import (
    MODEL_REPO, MODEL_REVISION, VAE_REPO, VAE_REVISION, UPSTREAM_COMMIT,
    _VAE_SOURCE_FILES, prepare, verify_conversion,
)
from .measure import DEFAULT_MEMORY_BUDGET_GIB, GPUExecution


def initial_noise(frames: int, seed: int) -> np.ndarray:
    """One request-local PCG64 FP32 draw; retained noise supports exact replay."""
    if type(frames) is not int or frames < 1 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Require positive frames and a 63-bit request seed")
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((frames, 64), dtype=np.float32)


@dataclass
class SongResult(ReferenceSongResult):
    noise: np.ndarray | None

    def save_artifacts(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Use an empty artifact directory to avoid mixing recordings")
        frames = len(self.semantic.tokens)
        expected_samples = 1920 * frames - 64
        if frames < 1:
            raise ValueError("A saved song must contain semantic codec frames")
        if not isinstance(self.config, dict) or not isinstance(self.config.get("generation"), dict):
            raise ValueError("Song configuration must retain effective generation settings")
        try:
            GenerationConfig.from_dict(self.config["generation"])
        except (TypeError, ValueError) as error:
            raise ValueError("Song generation configuration is invalid") from error
        if not isinstance(self.weights, dict) or not {"mot", "vae"} <= self.weights.keys():
            raise ValueError("Song weights must identify the generator and decoder")
        if (
            not isinstance(self.timing, dict)
            or self.timing.get("abc") != self.semantic.plan.timing
            or self.timing.get("semantic") != self.semantic.timing
        ):
            raise ValueError("Song timing must retain its planning and semantic stages")
        if (
            not isinstance(self.latents, np.ndarray)
            or self.latents.dtype != np.float32
            or self.latents.shape != (frames, 64)
            or not np.isfinite(self.latents).all()
        ):
            raise ValueError("Song latents must be finite FP32 [T,64]")
        if self.noise is not None and (
            not isinstance(self.noise, np.ndarray)
            or self.noise.dtype != np.float32
            or self.noise.shape != self.latents.shape
            or not np.isfinite(self.noise).all()
        ):
            raise ValueError("Song noise must be finite FP32 [T,64] when retained")
        if (
            self.sample_rate != 48000
            or not isinstance(self.audio, np.ndarray)
            or self.audio.dtype != np.float32
            or self.audio.shape != (expected_samples, 2)
            or not np.isfinite(self.audio).all()
        ):
            raise ValueError("Song audio must be finite FP32 stereo at its natural 48 kHz length")
        expected_identity = identity({
            "request": self.semantic.plan.request.to_dict(),
            "config": self.config,
            "weights": self.weights,
        })
        if self.request_identity != expected_identity:
            raise ValueError("Song identity does not match its request, configuration and weights")
        directory.mkdir(parents=True, exist_ok=True)
        if self.noise is not None:
            np.save(directory / "noise.npy", self.noise, allow_pickle=False)
        return super().save_artifacts(directory)


class YuE2Pipeline(ReferencePipeline):
    """Single-request pipeline. Weights may remain resident between serial calls.

    Inherited planning, semantic protocol, plan persistence and progress use the
    pinned upstream implementation. Only backend execution and model lifecycle
    are replaced. Concurrent calls on the same instance are unsupported.
    """

    def __init__(
        self, model_dir, vae_dir, *, precision="bf16", generation_config=None,
        memory_budget_gib=DEFAULT_MEMORY_BUDGET_GIB, vae_core_frames=256,
        query_chunk_size=256, progress=True, resource_path=None, require_ac=False,
    ):
        if platform.system() != "Darwin" or tuple(map(int, platform.mac_ver()[0].split(".")[:2])) < (26, 2):
            raise RuntimeError("The supported MLX M5 runtime requires macOS >=26.2")
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is required")
        for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
            if os.environ.get(name) == "1":
                raise RuntimeError(f"Unset {name}; fallback/fast-math is not a validated execution path")
        if precision not in {"bf16", "8bit", "4bit"}:
            raise ValueError("precision must be bf16, 8bit or 4bit")
        if type(progress) is not bool:
            raise TypeError("progress must be True or False")
        if type(vae_core_frames) is not int or vae_core_frames < 1:
            raise ValueError("vae_core_frames must be a positive integer")
        if query_chunk_size is not None and (type(query_chunk_size) is not int or query_chunk_size < 1):
            raise ValueError("query_chunk_size must be a positive integer")
        if generation_config is not None and not isinstance(generation_config, GenerationConfig):
            raise TypeError("generation_config must be a GenerationConfig")
        self.model_dir, self.vae_dir = Path(model_dir), Path(vae_dir)
        self.precision, self.progress = precision, progress
        self.backend, self.quantization = "mlx", precision
        self.device = mx.gpu
        self.memory_budget_gib = float(memory_budget_gib)
        self.vae_core_frames, self.query_chunk_size = vae_core_frames, query_chunk_size
        self.generation_config = generation_config or GenerationConfig()
        self.offload_ar = False
        with self._status("Verifying model files"):
            self.conversion = verify_conversion(self.model_dir)
            if not (self.model_dir / f"ar-{precision}.safetensors").is_file():
                raise FileNotFoundError(f"Convert the requested AR precision first: {precision}")
            vae_identity = model_identity(self.vae_dir)
            pinned_weight = _VAE_SOURCE_FILES["model.safetensors"]
            if (
                vae_identity["files"] == {"model.safetensors": pinned_weight}
                and vae_identity["config_sha256"] == _VAE_SOURCE_FILES["config.json"]["sha256"]
            ):
                vae_identity["source"] = {
                    "repository": VAE_REPO,
                    "revision": VAE_REVISION,
                    "revision_proof": "verified-pinned-content",
                }
            self.weights = {"mot": self.conversion, "vae": vae_identity}
        self._ar = self._bf16_ar = self._nar = self._vae = None
        # The reference decoder never sees a Torch generator model.
        self._model = None
        self.load_timing = {}
        self.runtime_sha256 = identity({
            "upstream_commit": UPSTREAM_COMMIT,
            "lyra": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "upstream": {p.name: sha256_file(p) for p in sorted(Path(__import__("yue2").__file__).parent.glob("*.py"))},
        })
        self.runtime = {name: version(name) for name in ("lyra-yue2", "mlx", "mlx-lm", "transformers", "numpy")}
        self.runtime.update(python=platform.python_version(), macos=platform.mac_ver()[0])
        self._closed = True
        self._gpu_execution = None
        with ExitStack() as resources:
            self._gpu_execution = resources.enter_context(
                GPUExecution(
                    backend="mlx", memory_budget_gib=self.memory_budget_gib,
                    resource_path=resource_path, require_ac=require_ac,
                )
            )
            resources.callback(self._release_models)
            mx.set_default_device(mx.gpu)
            self.tokenizer = YuE2TextTokenizer(self.model_dir / "qwen.tiktoken")
            self._resources = resources.pop_all()
        self._closed = False

    @classmethod
    def from_pretrained(
        cls, model=MODEL_REPO, *, vae=VAE_REPO, converted_dir="models/converted",
        revision=MODEL_REVISION, vae_revision=None,
        cache_dir=None, local_files_only=False, precision="bf16", progress=True,
        **kwargs,
    ):
        start = time.perf_counter()
        if str(vae) == VAE_REPO and vae_revision is None:
            vae_revision = VAE_REVISION
        model = Path(model).expanduser() if Path(model).expanduser().is_dir() else model
        saved = Path(model) / "pipeline.json"
        if saved.is_file():
            metadata = json.loads(saved.read_text())
            parent = saved.parent
            model = parent / metadata["model"]
            if vae == VAE_REPO:
                vae = parent / metadata["vae"]
            kwargs.setdefault("generation_config", GenerationConfig.from_dict(metadata["generation_config"]))
        with Progress(enabled=progress).stage("Resolving model files"):
            if (Path(model) / "conversion.json").is_file():
                model_path = Path(model)
            else:
                if str(model) == MODEL_REPO and revision != MODEL_REVISION:
                    raise ValueError("This port targets the pinned generator revision in PORTING.md")
                source = resolve_model(
                    model, revision=revision, cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                with GPUExecution(
                    backend="mlx",
                    memory_budget_gib=kwargs.get(
                        "memory_budget_gib", DEFAULT_MEMORY_BUDGET_GIB,
                    ),
                    require_ac=kwargs.get("require_ac", False),
                ):
                    model_path = prepare(
                        source, converted_dir, precision=precision,
                        source_revision=revision if str(model) == MODEL_REPO else None,
                    )
            if str(vae) == VAE_REPO and vae_revision != VAE_REVISION:
                raise ValueError("This port targets the pinned decoder revision in PORTING.md")
            vae_path = resolve_model(vae, revision=vae_revision, cache_dir=cache_dir, local_files_only=local_files_only)
        result = cls(model_path, vae_path, precision=precision, progress=progress, **kwargs)
        result.load_timing["resolve_conversion_integrity_seconds"] = time.perf_counter() - start
        return result

    def _check_execution(self):
        if self._closed:
            raise RuntimeError("YuE2Pipeline is closed")
        self._gpu_execution.check()

    def _guarded_cancelled(self, cancelled):
        def guarded():
            self._check_execution()
            return False if cancelled is None else cancelled()

        return guarded

    def plan(self, style=None, lyrics=None, *, tags=None, request=None, abc_sampling=None,
             cancelled=None, on_token=None, **kwargs):
        self._check_execution()
        result = super().plan(
            style, lyrics, tags=tags, request=request, abc_sampling=abc_sampling,
            cancelled=cancelled, on_token=on_token, **kwargs,
        )
        self._check_execution()
        return result

    def generate_semantic(self, plan, *, sampling=None, cancelled=None, on_token=None):
        self._check_execution()
        result = super().generate_semantic(
            plan, sampling=sampling, cancelled=cancelled, on_token=on_token,
        )
        self._check_execution()
        if not result.tokens:
            raise RuntimeError("Semantic generation produced no codec frames")
        return result

    def _record_load(self, stage, seconds):
        key = f"{stage}_load_events_seconds"
        events = (*self.load_timing.get(key, ()), seconds)
        self.load_timing[key] = events
        self.load_timing[f"{stage}_load_seconds"] = sum(events)

    def _load_model(self, for_nar=False):
        from .ar import load_ar
        from .nar import load_nar

        self._check_execution()
        if for_nar:
            if self.precision != "bf16" and self._ar is not None:
                self._ar = None
                mx.synchronize()
                mx.clear_cache()
                self._check_execution()
            if self._bf16_ar is None:
                if self.precision == "bf16" and self._ar is not None:
                    self._bf16_ar = self._ar
                else:
                    start = time.perf_counter()
                    with self._status("Loading BF16 acoustic conditioning"):
                        self._bf16_ar = load_ar(self.model_dir, precision="bf16", verify=False)
                        mx.eval(self._bf16_ar.parameters())
                    self._record_load("conditioning", time.perf_counter() - start)
                    self._check_execution()
            if self._nar is None:
                start = time.perf_counter()
                with self._status("Loading acoustic model"):
                    self._nar = load_nar(self.model_dir, self._bf16_ar, verify=False)
                    mx.eval(self._nar.parameters())
                self._record_load("nar", time.perf_counter() - start)
                self._check_execution()
            self._check_execution()
            return self._nar

        if (
            self.precision != "bf16"
            and self._ar is None
            and (self._bf16_ar is not None or self._nar is not None)
        ):
            self._nar = self._bf16_ar = None
            mx.synchronize()
            mx.clear_cache()
            self._check_execution()
        if self._ar is None:
            if self.precision == "bf16" and self._bf16_ar is not None:
                self._ar = self._bf16_ar
            else:
                start = time.perf_counter()
                with self._status(f"Loading {self.precision} AR model"):
                    self._ar = load_ar(self.model_dir, precision=self.precision, verify=False)
                    mx.eval(self._ar.parameters())
                self._record_load("ar", time.perf_counter() - start)
                self._check_execution()
        self._check_execution()
        return self._ar

    def _generate(self, prefix, sampling, seed, phase, **kwargs):
        from .ar import generate_tokens

        self._check_execution()
        if len(prefix) + sampling.max_tokens > CONTEXT:
            raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
        negative = kwargs.get("negative")
        if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
            raise ValueError("Negative prefix + generation budget exceeds context")
        cancelled = self._guarded_cancelled(kwargs.get("cancelled"))
        kwargs["cancelled"] = cancelled
        if cancelled():
            raise InterruptedError("Cancelled before prefill")
        model = self._load_model()
        callback = kwargs.pop("on_token", None)
        label = "Planning score" if phase == "abc" else "Generating song"
        with self._status(label, unit="tokens") as status:
            def on_output(token_phase, token):
                status.advance()
                if callback is not None:
                    callback(token_phase, token)
            observed = on_output if self.progress else callback
            result = generate_tokens(model, prefix, sampling, seed, phase, on_token=observed, **kwargs)
            if result[2]:
                status.finish(status="truncated")
        self._check_execution()
        return result

    def synthesize(self, semantic, *, cancelled=None, noise=None):
        from .nar import synthesize

        self._check_execution()
        if not isinstance(semantic, SemanticResult):
            raise TypeError("Pass the SemanticResult returned by generate_semantic()")
        if token_prefixes(semantic.plan.request, self.tokenizer, semantic.plan.abc_ids) != semantic.plan.prefix:
            raise ValueError("Semantic result does not retain the request's exact prefix")
        if not semantic.tokens:
            raise ValueError("Semantic result must contain at least one codec frame")
        cancelled = self._guarded_cancelled(cancelled)
        if cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        model = self._load_model(for_nar=True)
        with self._status("Synthesizing audio", unit="steps") as status:
            def report(completed, total):
                self._check_execution()
                if self.progress:
                    status.update(completed, total=total)

            result = synthesize(
                model, semantic.plan.prefix, semantic.tokens, semantic.plan.request.seed,
                steps=self.generation_config.ode_steps, context=self.generation_config.context,
                cancelled=cancelled, on_progress=report, noise=noise,
                query_chunk_size=self.query_chunk_size,
            )
        self._check_execution()
        return result

    def decode(self, latents, *, full=False, vae=None, cancelled=None):
        from .vae import load_decoder

        self._check_execution()
        with self._status("Loading MLX audio decoder"):
            if vae is not None:
                model = load_decoder(vae)
            else:
                if self._vae is None:
                    start = time.perf_counter()
                    self._vae = load_decoder(self.vae_dir)
                    self._record_load("vae", time.perf_counter() - start)
                model = self._vae
        with self._status("Decoding audio", unit="chunks") as status:
            def report(completed, total):
                self._check_execution()
                status.update(completed, total=total)
            result = model.decode(
                latents, full=full, core_frames=self.vae_core_frames,
                cancelled=self._guarded_cancelled(cancelled), on_progress=report,
            )
        self._check_execution()
        return np.clip(result, -1, 1)

    def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
        from .ar import _full_attention_requires_promotion
        config = super().effective_config(request, abc_sampling, semantic_sampling)
        config.update(
            vae_backend="mlx", backend="mlx", ar_precision=self.precision, kv_dtype="bfloat16",
            nar_dtype="bfloat16", conditioning_precision="bf16",
            attention_opmath="float32", mlx_tf32_enabled=False,
            attention_inputs="promoted_float32" if _full_attention_requires_promotion() else "native_steel_bfloat16",
            query_chunk_size=self.query_chunk_size, runtime=self.runtime,
            upstream_commit=UPSTREAM_COMMIT, execution_guard="GPUExecution",
            rng={"ar": "request_local_mlx", "acoustic": "numpy_pcg64_fp32_full_song_v1"},
        )
        return config

    def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                 semantic_sampling=None, cancelled=None, on_token=None, **kwargs):
        self._check_execution()
        request = self._request(style, lyrics, tags=tags, **kwargs)
        config = self.effective_config(request, abc_sampling, semantic_sampling)
        request_id = identity({"request": request.to_dict(), "config": config, "weights": self.weights})
        start = time.perf_counter()
        plan = self.plan(request=request, abc_sampling=abc_sampling, cancelled=cancelled, on_token=on_token)
        semantic = self.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token)
        noise = initial_noise(len(semantic.tokens), request.seed)
        nar_start = time.perf_counter()
        latents = self.synthesize(semantic, cancelled=cancelled, noise=noise)
        nar_seconds = time.perf_counter() - nar_start
        if self._guarded_cancelled(cancelled)():
            raise InterruptedError("Cancelled before VAE")
        vae_start = time.perf_counter()
        audio = self.decode(latents, cancelled=cancelled)
        timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                  "vae_seconds": time.perf_counter() - vae_start, "load": dict(self.load_timing),
                  "e2e_seconds": time.perf_counter() - start}
        Progress(enabled=self.progress).complete(len(audio) / 48000, timing["e2e_seconds"],
                                                truncated=plan.truncated or semantic.truncated)
        result = SongResult(
            audio, 48000, semantic, latents, config, self.weights, timing, request_id, noise,
        )
        self._check_execution()
        return result

    def save_pretrained(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("save_pretrained needs an empty pipeline destination")
        verify_conversion(self.model_dir)
        model_identity(self.vae_dir)
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.model_dir, directory / "generator")
        from yue2.storage import copy_model_files
        copy_model_files(self.vae_dir, directory / "vae")
        write_json(directory / "pipeline.json", {
            "model": "generator", "vae": "vae", "generation_config": self.generation_config.to_dict(),
            "source_weights": self.weights,
        })

    def _release_models(self):
        try:
            mx.synchronize()
        finally:
            self._ar = self._bf16_ar = self._nar = self._vae = self._model = None
            mx.clear_cache()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._resources.close()

    def __enter__(self):
        self._check_execution()
        return self

    def __exit__(self, *exc):
        if self._closed:
            return False
        self._closed = True
        if exc[0] is None:
            return self._resources.__exit__(*exc)
        try:
            self._resources.__exit__(*exc)
        except BaseException as cleanup_error:
            if exc[1] is not None:
                exc[1].add_note(f"Pipeline cleanup also failed: {cleanup_error}")
        return False
