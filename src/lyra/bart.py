"""Cached MLX BART decoder used by SheetSage2.

Architecture source: Transformers BartDecoder, as configured by the pinned
SheetSage2 modeling_sheetsage2.py. Inference disables dropout and uses GELU.
"""
import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj = nn.Linear(width, width), nn.Linear(width, width)
        self.v_proj, self.out_proj = nn.Linear(width, width), nn.Linear(width, width)

    def __call__(self, x, memory=None, cache=None, mask=None):
        b, t, width = x.shape
        dim = width // self.heads

        def split(value):
            return value.reshape(b, -1, self.heads, dim).transpose(0, 2, 1, 3)

        q = split(self.q_proj(x))
        if memory is not None and cache is not None:
            k, v = cache
        else:
            source = x if memory is None else memory
            k, v = split(self.k_proj(source)), split(self.v_proj(source))
            if cache is not None:
                k, v = mx.concatenate([cache[0], k], axis=2), mx.concatenate([cache[1], v], axis=2)
        attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-.5, mask=mask)
        return self.out_proj(attended.transpose(0, 2, 1, 3).reshape(b, t, width)), (k, v)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, heads = config["hidden_size"], config["num_attention_heads"]
        self.self_attn, self.encoder_attn = Attention(width, heads), Attention(width, heads)
        self.self_attn_layer_norm = nn.LayerNorm(width)
        self.encoder_attn_layer_norm = nn.LayerNorm(width)
        self.final_layer_norm = nn.LayerNorm(width)
        self.fc1, self.fc2 = nn.Linear(width, config["intermediate_size"]), nn.Linear(config["intermediate_size"], width)

    def __call__(self, x, memory, cache, mask):
        self_cache, cross_cache = cache if cache else (None, None)
        update, self_cache = self.self_attn(x, cache=self_cache, mask=mask)
        x = self.self_attn_layer_norm(x + update)
        update, cross_cache = self.encoder_attn(x, memory=memory, cache=cross_cache)
        x = self.encoder_attn_layer_norm(x + update)
        x = self.final_layer_norm(x + self.fc2(nn.gelu(self.fc1(x))))
        return x, (self_cache, cross_cache)


class BartDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_positions = nn.Embedding(config["max_output_seq_len"] + 2, config["hidden_size"])
        self.layernorm_embedding = nn.LayerNorm(config["hidden_size"])
        self.layers = [DecoderLayer(config) for _ in range(config["decoder_layers"])]

    def __call__(self, embedded, memory, cache=None):
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        positions = mx.arange(offset, offset + embedded.shape[1])
        x = self.layernorm_embedding(embedded + self.embed_positions(positions + 2))
        mask = mx.arange(offset + embedded.shape[1])[None, :] <= positions[:, None]
        updated = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, memory, None if cache is None else cache[i], mask)
            updated.append(state)
        return x, updated
