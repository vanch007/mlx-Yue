"""Isolate attention operand promotion without changing production code or limits."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import fidelity
import mlx.core as mx
import lyra.ar as ar
import lyra.nar as nar


def promoted(query, key, value, *, scale, mask=None):
    return mx.fast.scaled_dot_product_attention(query.astype(mx.float32), key.astype(mx.float32),
        value.astype(mx.float32), scale=scale, mask=mask).astype(query.dtype)


ar._source_sdpa = nar._source_sdpa = promoted
output = Path(sys.argv[sys.argv.index("--output") + 1])
try:
    fidelity.main()
finally:
    if output.exists():
        (output / "experiment.json").write_text(json.dumps({
            "scope": "Diagnostic override only; no production change or acceptance-limit change",
            "override": "All attention operands promoted to FP32; output rounded to source dtype",
        }, indent=2) + "\n")
