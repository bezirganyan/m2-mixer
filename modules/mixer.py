import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, hidden_dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            FeedForward(hidden_dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class FusionMixer(nn.Module):
    def __init__(self, hidden_dim, patch_size, image_size, num_mixers, token_dim, channel_dim,
                 dropout=0.):
        super().__init__()

        assert (image_size[0] % patch_size == 0) and (
                    image_size[1] % patch_size == 0), 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, hidden_dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_mixers):
            self.mixer_blocks.append(MixerBlock(hidden_dim, self.num_patch, token_dim, channel_dim, dropout=dropout))

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size, image_size, num_mixers, token_dim, channel_dim,
                 dropout=0.):
        super().__init__()

        assert (image_size[0] % patch_size == 0) and (image_size[1] % patch_size == 0), 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_mixers):
            self.mixer_blocks.append(MixerBlock(hidden_dim, self.num_patch, token_dim, channel_dim, dropout=dropout))

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        return x

class PNLPMixer(nn.Module):
    def __init__(self, max_seq_len, hidden_dim, num_mixers, mlp_hidden_dim,
                 dropout=0.):
        super().__init__()

        # hidden_dim = dim
        # self.num_patch = max_seq_len
        # seq_hidden_dim = token_dim
        # channel_hidden_dim = channel_dim


        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_mixers):
            self.mixer_blocks.append(MixerBlock(hidden_dim, max_seq_len, mlp_hidden_dim, mlp_hidden_dim, dropout=dropout))

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # self.mlp_head = nn.Sequential(
        #     nn.Linear(hidden_dim, num_classes)
        # )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        return x

if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                     hidden_dim=512, num_mixers=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
