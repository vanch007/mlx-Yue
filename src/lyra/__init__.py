"""Offline YuE2 generation on Apple Silicon."""
import os as _os

# MLX caches this switch on first use. Preserve explicit user settings so the
# guarded runtime can reject them rather than silently changing their policy.
_os.environ.setdefault("MLX_ENABLE_TF32", "0")


__version__ = "0.1.0"


def __getattr__(name):
    if name in {"YuE2Pipeline", "SongResult"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name in {"SymbolicPlan", "SemanticResult"}:
        from yue2 import pipeline

        return getattr(pipeline, name)
    if name in {"SongRequest", "Sampling", "GenerationConfig"}:
        from yue2 import protocol

        return getattr(protocol, name)
    raise AttributeError(name)
