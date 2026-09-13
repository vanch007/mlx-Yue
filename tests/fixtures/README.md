# Independent numerical fixtures

Generated locally using Python 3.12, PyTorch 2.11.0, Transformers 4.57.6,
torchaudio 2.11.0 and NumPy 2.2.6, FP32 evaluation mode, 2026-09-13.

- MERT: official `modeling_mert2.py` and `configuration_mert2.py` from
  m-a-p/MERT-v2-FullSong, revision `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`.
  Random seed 7; tiny architecture in config.json; input shape [1,1200].
  Captured frontend, convolutional subsampling and final Conformer output.
- BART: Transformers BartDecoder, seed 19, width 16, 2 layers, 2 heads,
  intermediate size 32, vocabulary 100, max positions 64, all dropout disabled.
  Captured decoder output with embedded input [1,8,16] and memory [1,10,16].
  Excluded unused embed_tokens weights because the runtime supplies embeddings.

These tiny fixtures verify architecture and caching, not pretrained-model
musical quality. Real-checkpoint evidence is recorded separately in reports.
