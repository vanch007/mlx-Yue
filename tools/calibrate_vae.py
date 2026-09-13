"""Independent CPU FP64 decoder anchor; never reads MLX outputs or changes production."""
import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from lyra.measure import GPUExecution
from yue2.modeling_vae import YuE2VAE
from yue2.storage import model_identity, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if version("torch") != "2.11.0":
        raise RuntimeError("Use the pinned .oracle runtime")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    latent = np.load(args.latents)
    if latent.ndim != 2 or len(latent) > 1100:
        raise ValueError("Expected up to 1100 latent frames [T,C]")
    write_json(args.output / "invocation.json", {
        "precision": "float64", "backend": "torch_cpu", "weights": model_identity(args.vae),
        "latents_sha256": sha256_file(args.latents), "script_sha256": sha256_file(__file__),
        "purpose": "independent_FP32_roundoff_anchor_not_production_decoder",
        "geometry": "64-frame cores, 16-frame exact convolution halo; assembled full output",
    })
    # Reuse the process/resource guard and serialization lock; all learned ops here are CPU FP64.
    with GPUExecution(backend="mps", memory_budget_gib=16, require_ac=True,
                      resource_path=args.output / "resources.jsonl") as guard, torch.inference_mode():
        model = YuE2VAE.from_pretrained(args.vae, decoder_only=True, device="cpu", local_files_only=True)
        model.decoder.double()
        total, pieces = model.natural_output_length(len(latent)), []
        for start in range(0, len(latent), 64):
            guard.check()
            end = min(start + 64, len(latent))
            left, right = max(0, start - 16), min(len(latent), end + 16)
            z = torch.from_numpy(latent[left:right].T[None]).double()
            tile = model.decoder(z).numpy()
            offset = (start - left) * 1920
            count = min(end * 1920, total) - start * 1920
            pieces.append(tile[..., offset:offset + count])
        guard.check()
        np.savez(args.output / "vae.npz", full=np.concatenate(pieces, axis=-1))


if __name__ == "__main__":
    main()
