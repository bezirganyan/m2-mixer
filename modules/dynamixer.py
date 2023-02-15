import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from modules.mixer import FeedForward


class DynaMixerOp(nn.Module):
    def __init__(self, dim, seq_len, num_head, reduced_dim=2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        B, L, C = x.shape
        weights = self.compress(x).reshape(B, L, self.num_head, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_head, -1)
        weights = self.generate(weights).reshape(B, self.num_head, L, L)
        weights = self.activation(weights)
        x = x.reshape(B, L, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(B, L, C)
        x = self.out(x)
        return x


class DynaMixerBlock(nn.Module):
    def __init__(self, hidden_dim, num_patch=7, num_head=8, reduced_dim=2, qkv_bias=False, dropout=0., **kwargs):
        # TODO - is num patch really resolution?
        super().__init__()
        self.resolution = num_patch
        self.num_head = num_head
        self.mix_h = DynaMixerOp(hidden_dim, num_patch, self.num_head, reduced_dim=reduced_dim)
        self.mix_w = DynaMixerOp(hidden_dim, num_patch, self.num_head, reduced_dim=reduced_dim)
        self.mlp_c = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.reweight = FeedForward(hidden_dim, hidden_dim // 4, out_dim=hidden_dim * 3)

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.mix_h(x.permute(0, 2, 1, 3).reshape(-1, H, C)).reshape(B, W, H, C).permute(0, 2, 1, 3)
        w = self.mix_w(x.reshape(-1, W, C)).reshape(B, H, W, C)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DynaMixer(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size, image_size, num_mixers, dropout=0., **kwargs):
        super().__init__()

        assert (image_size[0] % patch_size == 0) and (
                    image_size[1] % patch_size == 0), 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, patch_size, patch_size),
            Rearrange('b c h w -> b h w c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_mixers):
            self.mixer_blocks.append(DynaMixerBlock(hidden_dim, dropout=dropout, num_patch=(image_size[0] // patch_size)
                                                    , **kwargs))

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        return x


class FusionDynaMixer(nn.Module):
    def __init__(self, hidden_dim, num_patches, num_mixers, dropout=0., **kwargs):
        super().__init__()

        self.num_patch = num_patches
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_mixers):
            self.mixer_blocks.append(DynaMixerBlock(hidden_dim, dropout=dropout, num_patch=int(math.sqrt(num_patches)),
                                                    **kwargs))

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        return x