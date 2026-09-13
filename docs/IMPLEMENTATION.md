# Full local MLX implementation

The user approved implementing the audited missing functionality on 2026-09-13.

## Delivery sequence

1. Native FP32 MLX VAE with original checkpoint loading and matched decoder tests.
2. Torch-independent generator data/protocol/noise path; reproducibility policy recorded in each result.
3. Official CLI controls, serial batch, completed-result resume and diagnostics.
4. ABC inspection/comparison, listening artifacts and explicit MLX agent adapter.
5. Source-audio transcription and cover workflow using separately provisioned SheetSage2/MERT2, with backend and unsupported capabilities stated explicitly.
6. Real checkpoint generation, matched numerical checks and documented full-song evidence.

Pure MLX transcription is a separate model implementation from the YuE2-3B generator. Do not report a PyTorch transcription bridge as a native MLX transcription backend.

## Source references

- https://github.com/multimodal-art-projection/YuE/tree/88da114a67df892af0329472073b96a5ef700b93
- https://github.com/daig/yue2-mlx/tree/c0f0229df07daba14923627e9d78e084d82b7dc8
- https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html

Status: implementation in progress. Baseline and new acceptance results will be recorded under reports/ and summarized here.
