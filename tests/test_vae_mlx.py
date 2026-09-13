import mlx.core as mx
import numpy as np
import pytest
import torch

from lyra.vae import OobleckDecoder, decoder_weights
from yue2.modeling_vae import OobleckDecoder as TorchDecoder


@pytest.mark.parametrize("snake", [True, False])
def test_native_decoder_matches_official_convolutions(snake):
    torch.manual_seed(13)
    config = dict(out_channels=2, channels=4, latent_dim=3,
                  c_mults=[1, 2], strides=[3, 2], use_snake=snake)
    reference = TorchDecoder(**config).eval()
    native = OobleckDecoder(config)
    weights = {"decoder." + k: v.detach().numpy() for k, v in reference.state_dict().items()}
    native.load_weights(list(decoder_weights(weights).items()))
    z = np.random.default_rng(14).standard_normal((1, 3, 40)).astype(np.float32)
    with torch.no_grad():
        expected = reference(torch.from_numpy(z)).numpy().transpose(0, 2, 1)
    actual = np.array(native(mx.array(z.transpose(0, 2, 1))))
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=1e-4)


def test_native_decoder_halo_crop_and_cancellation():
    config = dict(out_channels=2, channels=2, latent_dim=3, c_mults=[1] * 6,
                  strides=[2, 2, 4, 4, 5, 6], use_snake=True, final_tanh=False)
    model = OobleckDecoder(config)
    z = np.random.default_rng(12).standard_normal((45, 3)).astype(np.float32)
    full = model.decode(z, full=True)
    tiled = model.decode(z, core_frames=17)
    assert full.shape == (1920 * len(z) - 64, 2)
    np.testing.assert_allclose(tiled, full, atol=2e-5, rtol=2e-4)
    with pytest.raises(InterruptedError):
        model.decode(z, cancelled=lambda: True)
