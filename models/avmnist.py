from abc import ABC, abstractmethod

from omegaconf import DictConfig

from models.train_test_module import AbstractTrainTestModule
from modules.gmpl import VisiongMLP, FusiongMLP
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


class AbstractAVMnistMixer(AbstractTrainTestModule, ABC):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AbstractAVMnistMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = None
        self.classifier = None

    def shared_step(self, batch):
        logits = self.get_logits(batch)
        labels = batch['label']
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    @abstractmethod
    def get_logits(self, batch):
        raise NotImplementedError

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]


class AVMnistImageMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistImageMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.model = MLPMixer(**model_cfg.modalities.image, dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        labels = batch['label']
        logits = self.model(image)
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistAudioMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistAudioMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.model = MLPMixer(**model_cfg.modalities.image, dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        audio = batch['audio']
        labels = batch['label']
        logits = self.model(audio)
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        # num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        num_patches = self.image_mixer.num_patch
        self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixerLF(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerLF, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        # num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        num_patches = self.image_mixer.num_patch
        # self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                  model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = torch.nn.Linear(model_cfg.modalities.classification.num_classes * 2,
                                                model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        # logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        image_logits = torch.softmax(self.classifier_image(image_logits.mean(dim=1)), dim=1)
        audio_logits = torch.softmax(self.classifier_audio(audio_logits.mean(dim=1)), dim=1)
        logits = self.classifier_fusion(torch.cat([image_logits, audio_logits], dim=1))
        # logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixerMultiLoss, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = MLPMixer(**image_config, dropout=dropout)
        self.audio_mixer = MLPMixer(**audio_config, dropout=dropout)
        num_patches = self.image_mixer.num_patch + self.audio_mixer.num_patch
        # num_patches = self.image_mixer.num_patch
        self.fusion_mixer = FusionMixer(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                  model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.image_criterion = CrossEntropyLoss()
        self.audio_criterion = CrossEntropyLoss()
        self.fusion_criterion = CrossEntropyLoss()

    def shared_step(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits.mean(dim=1))
        labels = batch['label']
        loss_image = self.image_criterion(image_logits, labels)
        loss_audio = self.audio_criterion(audio_logits, labels)
        loss_fusion = self.fusion_criterion(logits, labels)
        loss = loss_image + loss_audio + loss_fusion
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10))

        return [train_scores, val_scores, test_scores]


class AVMnistgMLP(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistgMLP, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        self.mute = model_cfg.get('mute', None)
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = VisiongMLP(**image_config, dropout=dropout)
        self.audio_mixer = VisiongMLP(**audio_config, dropout=dropout)
        num_patches = self.image_mixer.num_patch
        self.fusion_mixer = FusiongMLP(**multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.d_model,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)
        # logits = self.fusion_mixer(torch.cat([image_logits, audio_logits], dim=1))
        logits = self.fusion_mixer(torch.maximum(image_logits, audio_logits))
        logits = self.classifier(logits[:, 0])
        return logits
