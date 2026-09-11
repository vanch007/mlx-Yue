"""Source-precision behavior at autoregressive arithmetic boundaries."""
import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models import qwen3

from lyra.ar import SourceMLP


def test_swiglu_uses_fp32_silu_opmath_before_bf16_product():
    model = SourceMLP(qwen3.MLP(1, 1))
    for projection in (model.gate_proj, model.up_proj, model.down_proj):
        projection.weight = mx.ones((1, 1), dtype=mx.bfloat16)
    values = np.array([-16, -5, -2.71875, -.1875, 0, .1875, 2.71875, 5, 16],
                      dtype=np.float32)[:, None]
    source = torch.from_numpy(values).to(device="mps", dtype=torch.bfloat16)
    expected = (torch.nn.functional.silu(source) * source).float().cpu().numpy()
    actual = np.asarray(model(mx.array(values, dtype=mx.bfloat16)).astype(mx.float32))
    np.testing.assert_array_equal(actual, expected)
