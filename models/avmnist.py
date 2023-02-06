from omegaconf import DictConfig

from models.train_test_module import AbstractTrainTestModule
from modules.mixer import MLPool, MLPMixer, FusionMixer

import torch

from typing import List
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from torch import nn


class AVMnistImagePooler(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistImagePooler, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MLPool(**model_cfg.modalities.image)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dims[-1],
                                          model_cfg.modalities.classification.num_classes)

    def shared_step(self, batch):
        image = batch['image']
        labels = batch['label']
        logits = self.model(image)

        logits = self.classifier(logits.mean(dim=1))
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]


class AVMnistImageMixer(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistImageMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = MLPMixer(**model_cfg.modalities.image, dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def shared_step(self, batch):
        image = batch['image']
        labels = batch['label']
        logits = self.model(image)

        logits = self.classifier(logits.mean(dim=1))
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]

class AVMnistMixer(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        classification_config = model_cfg.modalities.classification
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.head = nn.Linear(multimodal_config.hidden_dim, classification_config.num_classes)

    def shared_step(self, batch):
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        logits = self.head(logits.mean(dim=1))
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]