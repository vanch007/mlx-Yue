"""Darwin whole-process footprint plus explicitly separate framework counters."""
from __future__ import annotations

import ctypes
import fcntl
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import psutil

DEFAULT_MEMORY_BUDGET_GIB = 16
_GIB = 2**30
_MIB = 2**20
_GPU_LOCK = threading.RLock()
_GPU_OWNERS = []
_GPU_LOCK_FD = None
_FP32_PRECISION_VERIFIED = False


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
        "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
        "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups",
        "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
        "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance",
        "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
        "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
        "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions",
        "cycles", "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time",
    )]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True) if platform.system() == "Darwin" else None
if _libproc is not None:
    _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libproc.proc_pid_rusage.restype = ctypes.c_int


class _VMStatistics64(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_uint32) for name in ("free", "active", "inactive", "wired")]
        + [(name, ctypes.c_uint64) for name in (
            "zero_fill", "reactivations", "pageins", "pageouts", "faults",
            "cow_faults", "lookups", "hits", "purges",
        )]
        + [(name, ctypes.c_uint32) for name in ("purgeable", "speculative")]
        + [(name, ctypes.c_uint64) for name in (
            "decompressions", "compressions", "swapins", "swapouts",
        )]
        + [(name, ctypes.c_uint32) for name in (
            "compressor", "throttled", "external", "internal",
        )]
        + [(name, ctypes.c_uint64) for name in ("uncompressed", "swapped")]
    )


if _libproc is not None:
    _system = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    _system.mach_host_self.restype = ctypes.c_uint32
    _system.host_statistics64.argtypes = (
        ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
    )
    _system.host_statistics64.restype = ctypes.c_int
    _system.sysctlbyname.argtypes = (
        ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p, ctypes.c_size_t,
    )
    _system.sysctlbyname.restype = ctypes.c_int
    _host = _system.mach_host_self()
    _page_bytes = os.sysconf("SC_PAGE_SIZE")


def _darwin_memory():
    # host_info.h: HOST_VM_INFO64=4, count is measured in 32-bit integers.
    statistics = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(statistics) // 4)
    status = _system.host_statistics64(_host, 4, ctypes.byref(statistics), ctypes.byref(count))
    if status or count.value * 4 < _VMStatistics64.uncompressed.offset + 8:
        raise OSError(f"host_statistics64 failed: status={status}, count={count.value}")
    pressure = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(pressure))
    if _system.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(pressure),
                           ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), "Reading memory pressure failed")
    return {
        # psutil's macOS sin/sout are pageins/pageouts, not compressor swap I/O.
        "system_swap_in_bytes": statistics.swapins * _page_bytes,
        "system_swap_out_bytes": statistics.swapouts * _page_bytes,
        "system_memory_pressure_level": pressure.value,
        "system_free_bytes": statistics.free * _page_bytes,
        "system_wired_bytes": statistics.wired * _page_bytes,
        "system_compressor_bytes": statistics.compressor * _page_bytes,
        "system_file_backed_bytes": statistics.external * _page_bytes,
    }

def power_source():
    raw = subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True).stdout.strip()
    return {"ac_connected": "'AC Power'" in raw, "pmset": raw}


def memory_snapshot():
    process = psutil.Process()
    swap = psutil.swap_memory()
    virtual = psutil.virtual_memory()
    result = {"rss_bytes": process.memory_info().rss, "system_swap_used_bytes": swap.used,
              "system_swap_in_bytes": swap.sin, "system_swap_out_bytes": swap.sout,
              "system_available_bytes": virtual.available,
              "system_total_bytes": virtual.total}
    if _libproc is not None:
        result.update(_darwin_memory())
        usage = _RUsageInfoV4()
        if _libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)):
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        result.update(physical_footprint_bytes=usage.phys_footprint,
                      process_lifetime_peak_footprint_bytes=usage.lifetime_max_phys_footprint)
    # sys.modules exposes partial imports. Keep sampling process memory without
    # touching incomplete backend APIs or blocking this thread on import locks.
    mx = sys.modules.get("mlx.core")
    mx_spec = getattr(mx, "__spec__", None)
    if mx_spec is not None and not getattr(mx_spec, "_initializing", False):
        result.update(mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                      mlx_peak_bytes=mx.get_peak_memory())
    torch = sys.modules.get("torch")
    torch_spec = getattr(torch, "__spec__", None)
    if (torch_spec is not None and not getattr(torch_spec, "_initializing", False)
            and torch.backends.mps.is_available()):
        result.update(mps_active_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


class ResourceMonitor:
    """Sample footprint and real swap I/O; flush evidence before checking limits."""

    def __init__(self, interval=0.25, require_ac=False, log_path=None, report_path=None,
                 on_sample=None, metadata=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Sampling interval must be finite and positive")
        self.interval, self.require_ac = interval, require_ac
        self.log_path = None if log_path is None else Path(log_path)
        self.report_path = None if report_path is None else Path(report_path)
        self.on_sample = on_sample
        self.metadata = {} if metadata is None else dict(metadata)
        self._stop = threading.Event()
        self.samples = []
        self.error = None
        self.power_start = self.power_end = None
        self._stream = self.thread = None
        self._closed = False
        self._owns_evidence = False
        self._exception = None
        self._sampled_power = None
        self._next_power_check = 5.0

    def _sample(self):
        sample = {"elapsed_seconds": time.perf_counter() - self.start, **memory_snapshot()}
        if self.require_ac:
            if sample["elapsed_seconds"] >= self._next_power_check:
                self._sampled_power = power_source()
                self._next_power_check = sample["elapsed_seconds"] + 5.0
            sample["ac_connected"] = self._sampled_power["ac_connected"]
        self.samples.append(sample)
        if self._stream is not None:
            self._stream.write(json.dumps(sample) + "\n")
            self._stream.flush()
        if self.require_ac and not sample["ac_connected"]:
            raise RuntimeError("AC power disconnected during acceptance execution")
        if self.on_sample is not None:
            self.on_sample(sample)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception as error:
                if self.error is None:
                    self.error = error
                # Keep recording while the current GPU operation unwinds.

    def __enter__(self):
        if self._closed or self.thread is not None:
            raise RuntimeError("Resource monitors are single-use")
        self.start = time.perf_counter()
        try:
            for path in (self.log_path, self.report_path):
                if path is not None and path.exists():
                    raise FileExistsError(f"Resource evidence already exists: {path}")
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.log_path.open("x", encoding="utf-8")
            self._owns_evidence = True
            self.power_start = power_source()
            self._sampled_power = self.power_start
            if self.require_ac and not self.power_start["ac_connected"]:
                raise RuntimeError("AC power is required for an acceptance benchmark")
            self._sample()
            self.thread = threading.Thread(target=self._run, name="lyra-resources", daemon=True)
            self.thread.start()
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self._closed:
            return False
        self._closed = True
        self._stop.set()
        if self.thread is not None:
            self.thread.join()
        self._exception = None if exc_value is None else f"{exc_type.__name__}: {exc_value}"
        try:
            self._sample()
        except Exception as error:
            if self.error is None:
                self.error = error
        try:
            self.power_end = power_source()
            if self.require_ac and not self.power_end["ac_connected"]:
                raise RuntimeError("AC power disconnected during acceptance benchmark")
        except Exception as error:
            if self.error is None:
                self.error = error
        finally:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            if self.report_path is not None and self._owns_evidence:
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                with self.report_path.open("x", encoding="utf-8") as destination:
                    json.dump(self.report(), destination, indent=2)
                    destination.write("\n")
        if self.error is not None and exc_type is None:
            raise self.error
        return False

    def report(self):
        samples = list(self.samples)
        byte_keys = {key for sample in samples for key in sample if key.endswith("_bytes")}
        return {
            "memory_counter_schema": 2,
            "swap_counter_source": "host_statistics64.swapins/swapouts" if _libproc else "psutil",
            "sampling_interval_seconds": self.interval,
            "power_sampling_interval_seconds": 5.0 if self.require_ac else None,
            "metadata": self.metadata,
            "power_start": self.power_start, "power_end": self.power_end,
            "start": samples[0] if samples else None, "end": samples[-1] if samples else None,
            "sampled_maxima": {key: max(s.get(key, 0) for s in samples) for key in sorted(byte_keys)},
            "monitor_error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
            "exception": self._exception,
            "samples": samples,
        }


class GPUExecution:
    """Serialize Lyra GPU residents and fail closed on unsafe memory demand.

    Allocator settings are process-global and remain conservative after exit.
    Whole-process and system checks are sampled; callers check the latched error
    between operations. MPS has a hard allocation cap. MLX's limit is advisory:
    its operations must also use bounded prefill/query tiles.
    Existing idle swap is not a failure, but more than 64 MiB of new swap-outs is.
    """

    def __init__(self, backend="mlx", memory_budget_gib=DEFAULT_MEMORY_BUDGET_GIB,
                 resource_path=None, require_ac=False):
        if backend not in {"mlx", "mps"}:
            raise ValueError("GPU backend must be mlx or mps")
        minimum = 5 if backend == "mlx" else 1
        if (isinstance(memory_budget_gib, bool) or not math.isfinite(memory_budget_gib)
                or not minimum < memory_budget_gib <= psutil.virtual_memory().total / _GIB - 4):
            raise ValueError(f"Memory budget must exceed {minimum} GiB and leave 4 GiB OS headroom")
        self.backend, self.memory_budget_gib = backend, float(memory_budget_gib)
        self._entered = False
        self._baseline = None
        path = None if resource_path is None else Path(resource_path)
        if path is not None and path.suffix != ".jsonl":
            raise ValueError("GPU resource_path must end in .jsonl")
        self.monitor = ResourceMonitor(
            require_ac=require_ac, log_path=path,
            report_path=None if path is None else path.with_suffix(".json"),
            on_sample=self._check_sample,
            metadata={"gpu_backend": backend, "memory_budget_gib": self.memory_budget_gib,
                      "whole_process_limit_kind": "sampled",
                      "maximum_new_swap_out_bytes": 64 * _MIB},
        )

    def _check_sample(self, sample):
        if self._baseline is None:
            self._baseline = sample
        footprint = sample["physical_footprint_bytes"]
        if footprint > self.memory_budget_gib * _GIB:
            raise MemoryError(f"Process footprint {footprint / _GIB:.2f} GiB exceeds "
                              f"{self.memory_budget_gib:g} GiB budget")
        pressure = sample["system_memory_pressure_level"]
        if pressure != 1:
            raise MemoryError(f"System memory pressure is not normal (level={pressure})")
        if sample["system_available_bytes"] < 2 * _GIB:
            raise MemoryError("Less than 2 GiB of available system memory remains")
        swapped = sample["system_swap_out_bytes"] - self._baseline["system_swap_out_bytes"]
        growth = sample["system_swap_used_bytes"] - self._baseline["system_swap_used_bytes"]
        if swapped > 64 * _MIB or growth > 128 * _MIB:
            raise MemoryError(f"Stopping GPU workload after new swapping: "
                              f"{swapped / _MIB:.1f} MiB out, {growth / _MIB:.1f} MiB used growth")

    def _apply_limits(self):
        global _FP32_PRECISION_VERIFIED
        budget = min(owner.memory_budget_gib for owner in _GPU_OWNERS)
        mps_limit = int((budget - 1) * _GIB)
        torch = sys.modules.get("torch")
        mx = sys.modules.get("mlx.core")
        if self.backend == "mlx":
            if mx is None:
                raise RuntimeError("Import MLX before entering its GPU execution guard")
            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal is required; CPU fallback is not supported")
            mx.set_default_device(mx.gpu)
            self.monitor.metadata["mlx_default_device"] = str(mx.default_device())
            mx.set_memory_limit(int((budget - 5) * _GIB))
            mx.set_cache_limit(128 * _MIB)
            self.monitor.metadata["mlx_advisory_memory_limit_bytes"] = int((budget - 5) * _GIB)
            self.monitor.metadata["mlx_cache_limit_bytes"] = 128 * _MIB
            if os.environ.get("MLX_ENABLE_TF32") != "0":
                raise RuntimeError("Set MLX_ENABLE_TF32=0 before using MLX for faithful attention")
            if not _FP32_PRECISION_VERIFIED:
                # The environment flag is cached inside MLX and has no public
                # getter. Detect an earlier TF32 initialization once per process.
                operand = mx.full((128, 128), 1 + 2**-14, dtype=mx.float32)
                product = operand @ mx.eye(128, dtype=mx.float32)
                if mx.max(mx.abs(product - operand)).item() != 0:
                    raise RuntimeError(
                        "MLX was initialized with reduced FP32 precision; restart the process "
                        "with MLX_ENABLE_TF32=0 before any MLX computation"
                    )
                _FP32_PRECISION_VERIFIED = True
            self.monitor.metadata["mlx_tf32_enabled"] = False
            self.monitor.metadata["mlx_fp32_precision_verified"] = True
        elif torch is None:
            raise RuntimeError("Import Torch before entering its GPU execution guard")
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(mps_limit / torch.mps.recommended_max_memory())
            self.monitor.metadata["mps_hard_limit_bytes"] = mps_limit
            self.monitor.metadata["mps_hard_limit_scope"] = "all_process_metal_at_mps_allocation"

    def __enter__(self):
        global _GPU_LOCK_FD
        if self._entered:
            raise RuntimeError("This GPU execution context is already active")
        if _libproc is None:
            raise RuntimeError("The guarded GPU runtime requires macOS")
        if not _GPU_LOCK.acquire(blocking=False):
            raise RuntimeError("Another thread owns Lyra GPU execution")
        try:
            if _GPU_OWNERS and _GPU_OWNERS[-1].backend != self.backend:
                raise RuntimeError("Nested GPU execution must use the same backend budget")
            if not _GPU_OWNERS:
                path = Path(tempfile.gettempdir()) / f"lyra-gpu-{os.getuid()}.lock"
                descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    os.close(descriptor)
                    raise RuntimeError("Another Lyra process owns the GPU; run workloads serially") from error
                _GPU_LOCK_FD = descriptor
            _GPU_OWNERS.append(self)
            self._entered = True
            self.monitor.__enter__()
            self._apply_limits()
            return self
        except BaseException as error:
            if self._entered:
                self.__exit__(type(error), error, error.__traceback__)
            else:
                _GPU_LOCK.release()
            raise

    def check(self):
        if not self._entered:
            raise RuntimeError("GPU execution context is closed")
        for owner in _GPU_OWNERS:
            if owner.monitor.error is not None:
                error = owner.monitor.error
                # Do not attach model-call tracebacks to the resident monitor.
                raise type(error)(*error.args) from error

    def report(self):
        return {**self.monitor.report(), "gpu_backend": self.backend,
                "memory_budget_gib": self.memory_budget_gib,
                "maximum_new_swap_out_bytes": 64 * _MIB}

    def __exit__(self, exc_type, exc_value, traceback):
        global _GPU_LOCK_FD
        if not self._entered:
            return False
        try:
            return self.monitor.__exit__(exc_type, exc_value, traceback)
        finally:
            self._entered = False
            _GPU_OWNERS.remove(self)
            if not _GPU_OWNERS:
                os.close(_GPU_LOCK_FD)
                _GPU_LOCK_FD = None
            _GPU_LOCK.release()

    def close(self):
        self.__exit__(None, None, None)
