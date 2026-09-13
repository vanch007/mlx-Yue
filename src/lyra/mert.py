"""MLX MERT2 frontend and convolutional subsampling (official checkpoint layout).

Source: https://huggingface.co/m-a-p/MERT-v2-FullSong/blob/d8ba1c745e733b3908ce6ad16ebeb17ac7600a42/modeling_mert2.py
"""
import mlx.core as mx
import mlx.nn as nn


def apply_layers(layers, x):
    for layer in layers:
        x = apply_layers(layer, x) if isinstance(layer, list) else layer(x)
    return x


class MelFrontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_fft, self.hop = config["n_fft"], config["hop_length"]
        bins = config["num_mel_bins"]
        self.mel_mean, self.mel_std = mx.zeros((bins,)), mx.ones((bins,))
        self.spectrogram = {"window": mx.zeros((config["win_length"],))}
        self.mel_scale = {"fb": mx.zeros((self.n_fft // 2 + 1, bins))}

    def __call__(self, waveform):
        # torchaudio Spectrogram defaults: centered reflect padding, periodic Hann,
        # unnormalized power spectrum. Window and mel filterbank are checkpoint buffers.
        x = mx.pad(waveform.astype(mx.float32), [(0, 0), (self.n_fft // 2,) * 2], mode="reflect")
        n = (x.shape[1] - self.n_fft) // self.hop + 1
        frames = mx.as_strided(x, (x.shape[0], n, self.n_fft), (x.shape[1], self.hop, 1))
        window = self.spectrogram["window"]
        if window.size != self.n_fft:
            left = (self.n_fft - window.size) // 2
            window = mx.pad(window, (left, self.n_fft - window.size - left))
        spec = mx.fft.rfft(frames * window, axis=-1)
        power = spec.real ** 2 + spec.imag ** 2
        mel = 10 * mx.log10(mx.maximum(power @ self.mel_scale["fb"], 1e-10))
        return (mel[:, :-1] - self.mel_mean) / mx.maximum(self.mel_std, 1e-5)


class GlobalResponseNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight, self.bias = mx.zeros((1, 1, width)), mx.zeros((1, 1, width))

    def __call__(self, x):
        magnitude = mx.sqrt(mx.sum(x * x, axis=1, keepdims=True))
        normalized = magnitude / (mx.mean(magnitude, axis=-1, keepdims=True) + 1e-6)
        return self.weight * x * normalized + self.bias + x


class ConvNextLayer(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.depthwise_block = [nn.Identity(), nn.Conv1d(width, width, 7, padding=3, groups=width), nn.Identity()]
        self.pointwise_block = [nn.LayerNorm(width, eps=eps), nn.Linear(width, 4 * width),
                               nn.GELU(), GlobalResponseNorm(4 * width), nn.Linear(4 * width, width)]

    def __call__(self, x):
        return x + apply_layers(self.pointwise_block, apply_layers(self.depthwise_block, x))


class ConvNextBlock(nn.Module):
    def __init__(self, cin, cout, stride, depth, eps):
        super().__init__()
        self.resampling_layer = ([nn.LayerNorm(cin, eps=eps), nn.Identity(),
                                  nn.Conv1d(cin, cout, 2, stride=stride), nn.Identity()]
                                 if cin != cout or stride > 1 else [])
        self.convnext_layers = [ConvNextLayer(cout, eps) for _ in range(depth)]

    def __call__(self, x):
        return apply_layers(self.convnext_layers, apply_layers(self.resampling_layer, x))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.base = config["rotary_embedding_base"]
        self.query_proj, self.key_proj = nn.Linear(width, width), nn.Linear(width, width)
        self.value_proj, self.out_proj = nn.Linear(width, width), nn.Linear(width, width)

    def __call__(self, x):
        b, t, width = x.shape
        dim = width // self.heads
        shape = (b, t, self.heads, dim)
        q, k, v = [p(x).reshape(shape).transpose(0, 2, 1, 3)
                   for p in (self.query_proj, self.key_proj, self.value_proj)]
        q = mx.fast.rope(q, dims=dim, traditional=False, base=self.base, scale=1., offset=0)
        k = mx.fast.rope(k, dims=dim, traditional=False, base=self.base, scale=1., offset=0)
        # Queries are tiled; every query retains the full song's keys/values.
        parts = [mx.fast.scaled_dot_product_attention(q[:, :, i:i + 256], k, v, scale=dim**-.5)
                 for i in range(0, t, 256)]
        y = mx.concatenate(parts, axis=2).transpose(0, 2, 1, 3).reshape(b, t, width)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.w_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def __call__(self, x):
        return self.w_2(nn.gelu(self.w_1(x)))


def glu(x):
    first, second = mx.split(x, 2, axis=-1)
    return first * mx.sigmoid(second)


class ConvolutionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, kernel = config["hidden_size"], config["conv_depthwise_kernel_size"]
        self.layer_norm = nn.LayerNorm(width, eps=config["layer_norm_eps"])
        self.conv_block = [nn.Identity(), nn.Conv1d(width, 2 * width, 1, bias=False), glu,
                           nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2, groups=width, bias=False),
                           [nn.Identity(), nn.LayerNorm(width, eps=config["layer_norm_eps"]), nn.Identity()],
                           nn.GELU(), nn.Conv1d(width, width, 1, bias=False), nn.Identity()]

    def __call__(self, x):
        return apply_layers(self.conv_block, self.layer_norm(x))


class ConformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ffn1_layer_norm, self.attn_layer_norm = nn.LayerNorm(width, eps=eps), nn.LayerNorm(width, eps=eps)
        self.ffn2_layer_norm, self.final_layer_norm = nn.LayerNorm(width, eps=eps), nn.LayerNorm(width, eps=eps)
        self.ffn1, self.ffn2 = FeedForward(config), FeedForward(config)
        self.attn, self.conv_module = Attention(config), ConvolutionModule(config)

    def __call__(self, x):
        x = x + .5 * self.ffn1(self.ffn1_layer_norm(x))
        x = x + self.attn(self.attn_layer_norm(x))
        x = x + self.conv_module(x)
        x = x + .5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class MERT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = MelFrontend(config)
        channels = [config["num_mel_bins"], *config["subsampling_channels"]]
        self.subsampling_module = [ConvNextBlock(channels[i], channels[i + 1], (1, 2, 2)[i],
                                   config["subsampling_depths"][i], config["subsampling_layer_norm_eps"])
                                   for i in range(3)]
        self.layers = [ConformerBlock(config) for _ in range(config["num_hidden_layers"])]

    def __call__(self, waveform, layer_weights=None, cancelled=None):
        x = apply_layers(self.subsampling_module, self.feature_extractor(waveform))
        mixed = None if layer_weights is None else x * layer_weights[0]
        for i, layer in enumerate(self.layers):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during MERT2 encoding")
            x = layer(x)
            if mixed is not None:
                mixed = mixed + x * layer_weights[i + 1]
            mx.eval(x, mixed)
        return x if mixed is None else mixed


def convert_mert_weights(weights):
    result = []
    for key, value in weights.items():
        if value.ndim == 3 and key.endswith("weight") and "pointwise_block.3." not in key:
            value = value.transpose(0, 2, 1)
        result.append((key, mx.array(value)))
    return result
