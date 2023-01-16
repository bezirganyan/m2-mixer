from typing import List

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import MSELoss

from models.train_test_module import AbstractTrainTestModule
from modules.gmpl import GatingMlpBlock


class gMLPEncoderDecoder(nn.Module):
    def __init__(
            self,
            d_ffn,
            hidden_dims,
            seq_len,
            n_blocks,
            prob_0_L=(1, 0.5),
            dropout=0.
    ):
        super().__init__()

        self.survival_probs = torch.linspace(prob_0_L[0], prob_0_L[1], n_blocks)
        self.blocks = nn.ModuleList([])
        prev_dim = hidden_dims[0]
        for i in range(n_blocks):
            self.blocks.append(
                GatingMlpBlock(prev_dim, d_ffn, seq_len, self.survival_probs[i], hidden_dims[i], dropout))
            prev_dim = hidden_dims[i]

    def forward(self, x):
        for gmlp_block in self.blocks:
            x = gmlp_block(x)
        return x


class GMLPAutoencoder(AbstractTrainTestModule):
    def __init__(
            self,
            model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs
    ):
        super().__init__(optimizer_cfg, **kwargs)
        image_cfg = model_cfg.modalities.image

        assert (image_cfg.image_size[0] % image_cfg.patch_size == 0) and (
                    image_cfg.image_size[1] % image_cfg.patch_size == 0), 'Image dimensions must be divisible by the patch size.'
        self.image_size = image_cfg.image_size
        self.n_patches = (image_cfg.image_size[0] // image_cfg.patch_size) * (image_cfg.image_size[1] // image_cfg.patch_size)
        self.patch_size = image_cfg.patch_size
        # self.seq_len = self.n_patches + 1
        self.seq_len = self.n_patches

        d_model = image_cfg.hidden_dims[0]
        self.patch_embedding = nn.Linear(image_cfg.n_channels * image_cfg.patch_size ** 2, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder = gMLPEncoderDecoder(image_cfg.d_ffn, image_cfg.hidden_dims,
                                          self.seq_len, image_cfg.n_blocks, image_cfg.prob_0_L, model_cfg.dropout)
        self.decoder = gMLPEncoderDecoder(image_cfg.d_ffn, image_cfg.hidden_dims[::-1],
                                          self.seq_len, image_cfg.n_blocks, image_cfg.prob_0_L, model_cfg.dropout)

    def forward(self, x):
        n_samples = x.shape[0]

        x = einops.rearrange(
            x, "n c (h p1) (w p2) -> n (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size
        )
        x = self.patch_embedding(x)

        # cls_token = self.cls_token.expand(n_samples, 1, -1)
        # x = torch.cat((cls_token, x), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        # cls_token_final = x[:, 0]
        x = einops.rearrange(
            x, "n h (c w) -> n c h w", h=self.image_size[0], w=self.image_size[1]
        )
        # x = x.reshape(x.shape[0], self.in_channels, *self.image_size)
        return x

    def setup_criterion(self) -> torch.nn.Module:
        return MSELoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        return None, None, None

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.forward(image.float())
        loss = self.criterion(logits, image.float())

        return {
            'preds': None,
            'labels': None,
            'loss': loss
        }


if __name__ == "__main__":
    img = torch.ones([1, 3, 160, 256])

    encoder = gMLPEncoderDecoder(3, [512, 256, 128], 16, (160, 256), 2, 256, 256, dropout=0.1)
    decoder = gMLPEncoderDecoder(3, [128, 256, 512], 16, (160, 256), 2, 256, 256, dropout=0.1)
    out_img = decoder(encoder(img))

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
