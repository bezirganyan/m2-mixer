import torch
from torch import nn
from modules.mixer import MLPMixer, PNLPMixer, FusionMixer


class MultimodalMixer(nn.Module):
    def __init__(self, image_config, text_config, multimodal_config, bottleneck_config, clasification_config,
                 dropout=0.):
        super().__init__()
        self.bottleneck = nn.Linear((2 * bottleneck_config.window_size + 1) * bottleneck_config.feature_size,
                                    bottleneck_config.hidden_dim)
        self.text_mixer = PNLPMixer(**text_config, dropout=dropout)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.fusion_mixer = FusionMixer(**multimodal_config, dropout=dropout)
        self.head = nn.Linear(multimodal_config.hidden_dim, clasification_config.num_classes)

    def forward(self, image, text):
        features = self.bottleneck(text)
        text_reprs = self.text_mixer(features)

        image_reprs = self.image_mixer(image.float())

        multimodal_reprs = torch.cat((text_reprs, image_reprs), dim=1).unsqueeze(1)
        # multimodal_reprs = (text_reprs + image_reprs).unsqueeze(1)
        multimodal_reprs = self.fusion_mixer(multimodal_reprs)

        logits = self.head(multimodal_reprs.mean(dim=1))
        return logits


class MosiMixer(nn.Module):
    def __init__(self, visual_config, text_config, audio_config, multimodal_config, bottleneck_config, clasification_config,
                 dropout=0.):
        super().__init__()
        self.bottleneck = nn.Linear((2 * bottleneck_config.window_size + 1) * bottleneck_config.feature_size,
                                    bottleneck_config.hidden_dim)
        self.text_mixer = PNLPMixer(**text_config, dropout=dropout)
        self.visual_mixer = MLPMixer(**visual_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        self.fusion_mixer = FusionMixer(**multimodal_config, dropout=dropout)
        self.head = nn.Linear(multimodal_config.hidden_dim, clasification_config.num_classes)

    def forward(self, image, text, audio):
        features = self.bottleneck(text)
        text_reprs = self.text_mixer(features)

        visual_reprs = self.visual_mixer(image.float().unsqueeze(1))
        audio_reprs = self.audio_mixer(audio.float().unsqueeze(1))

        multimodal_reprs = torch.cat((text_reprs, visual_reprs, audio_reprs), dim=1).unsqueeze(1)
        # multimodal_reprs = (text_reprs + visual_reprs).unsqueeze(1)
        multimodal_reprs = self.fusion_mixer(multimodal_reprs)

        logits = self.head(multimodal_reprs.mean(dim=1))
        return logits
