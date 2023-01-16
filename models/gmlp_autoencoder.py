from typing import List

import torch
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from torchmetrics import MeanSquaredError, F1Score

from models.train_test_module import AbstractTrainTestModule
from modules.mixer import MixerBlock, MLPMixer


class ScaleForward(nn.Module):
    def __init__(self, dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, patch_size, image_size, num_mixers, token_dim, channel_dim,
                 dropout=0.):
        super().__init__()
        super().__init__()

        assert (image_size[0] % patch_size == 0) and (
                image_size[1] % patch_size == 0), 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        prev_dim = hidden_dims[0]
        for hd in hidden_dims:
            self.mixer_blocks.append(ScaleForward(prev_dim, hd, dropout))
            prev_dim = hd
            for _ in range(num_mixers):
                self.mixer_blocks.append(MixerBlock(hd, self.num_patch, token_dim, channel_dim, dropout=dropout))

        self.layer_norm = nn.LayerNorm(hidden_dims[-1])

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        return x


class MixerDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, patch_size, image_size, num_mixers, token_dim, channel_dim,
                 dropout=0.):
        super().__init__()
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        assert (image_size[0] % patch_size == 0) and (
                image_size[1] % patch_size == 0), 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_image = nn.Sequential(
            # nn.Conv2d(in_channels, hidden_dims[0], patch_size, patch_size),
            Rearrange('b (h w) c -> b c h w', h=image_size[0] // patch_size, w=image_size[1] // patch_size),
        )

        self.mixer_blocks = nn.ModuleList([])

        hidden_dims.append(in_channels * patch_size * patch_size)
        prev_dim = hidden_dims[0]
        for hd in hidden_dims:
            self.mixer_blocks.append(ScaleForward(prev_dim, hd, dropout))
            prev_dim = hd
            for _ in range(num_mixers):
                self.mixer_blocks.append(MixerBlock(hd, self.num_patch, token_dim, channel_dim, dropout=dropout))

        self.layer_norm = nn.LayerNorm(hidden_dims[-1])

    def forward(self, x):

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        # restore to image
        x = self.to_image(x)
        x = x.reshape(x.shape[0], self.in_channels, *self.image_size)
        return x


class MixerAutoencoder(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MixerAutoencoder, self).__init__(optimizer_cfg, **kwargs)
        self.encoder = MixerEncoder(dropout=model_cfg.dropout, **model_cfg.modalities.image)
        model_cfg.modalities.image.hidden_dims = model_cfg.modalities.image.hidden_dims[::-1]
        self.decoder = MixerDecoder(dropout=model_cfg.dropout, **model_cfg.modalities.image)

    def setup_criterion(self) -> torch.nn.Module:
        return MSELoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        return None, None, None

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.encoder(image.float())
        logits = self.decoder(logits)
        loss = self.criterion(logits, image.float())

        return {
            'preds': None,
            'labels': None,
            'loss': loss
        }


if __name__ == "__main__":
    img = torch.ones([1, 3, 160, 256])

    encoder = MixerEncoder(3, [512, 256, 128], 16, (160, 256), 2, 256, 256, dropout=0.1)
    decoder = MixerDecoder(3, [128, 256, 512], 16, (160, 256), 2, 256, 256, dropout=0.1)
    out_img = decoder(encoder(img))

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]


class MMIMDBEncoderClassifier(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(MMIMDBEncoderClassifier, self).__init__(optimizer_cfg, **kwargs)
        self.encoder = MixerEncoder(dropout=model_cfg.dropout, **model_cfg.modalities.image)
        autoencoder = MixerAutoencoder.load_from_checkpoint(model_cfg.pretrained_autoencoder_path, model_cfg=model_cfg,
                                                            optimizer_cfg=optimizer_cfg)
        self.encoder.load_state_dict(autoencoder.encoder.state_dict())
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(model_cfg.modalities.image.hidden_dims[0] * self.encoder.num_patch, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, model_cfg.modalities.classification.num_classes))

    def setup_criterion(self) -> torch.nn.Module:
        return BCEWithLogitsLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        val_score = F1Score(task="multilabel", num_labels=23, average='weighted')
        test_score = F1Score(task="multilabel", num_labels=23, average='weighted')

        return [train_score, val_score, test_score]

    def shared_step(self, batch):
        image = batch['image']
        text = batch['text']
        labels = batch['label']
        logits = self.encoder(image.float())
        logits = self.classifier(logits.reshape(logits.shape[0], -1))
        loss = self.criterion(logits, labels.float())
        preds = torch.sigmoid(logits) > 0.5

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }


if __name__ == "__main__":
    img = torch.ones([1, 3, 160, 256])

    encoder = MixerEncoder(3, [512, 256, 128], 16, (160, 256), 2, 256, 256, dropout=0.1)
    decoder = MixerDecoder(3, [128, 256, 512], 16, (160, 256), 2, 256, 256, dropout=0.1)
    out_img = decoder(encoder(img))

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
